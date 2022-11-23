from functools import lru_cache
from typing import Iterable, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.fft import irfftn, rfftn

from .utils import to_ntuple


@lru_cache(1024)
def opt_blocksize(input_size, kernel_size, speed_impt=0.5):
    def prod(X):
        result = X[0]
        for x in X[1:]:
            result = result * x
        return result

    kernel_cost = []
    signal_cost = []
    logs = []
    for i, (s, k) in enumerate(zip(input_size[2:], kernel_size[2:])):
        shape = [1] * (len(input_size) - 2)
        shape[i] = -1
        padding = k // 2 * 2
        block_size = np.arange(1, s + 1).reshape(*shape)
        block_count = (s - 1) // block_size + 1
        log = np.ceil(np.log2(block_size + padding))
        kernel_cost.append(2**log)
        signal_cost.append(block_count * 2**log)
        logs.append(log + 1)
    memo = prod(input_size[:2]) * prod(signal_cost)
    memo += prod(kernel_size[:2]) * prod(kernel_cost)
    time = memo * sum(logs)
    tradoff = time ** (speed_impt) * memo ** (1 - speed_impt)
    index = np.argmin(tradoff)
    block_size = []
    for s in input_size[2:][::-1]:
        block_size.append(index % s + 1)
        index //= s
    return block_size[::-1]


def complex_matmul(a: Tensor, b: Tensor, groups: int = 1) -> Tensor:
    return torch.einsum(
        "bgi...f,goi...->bgo...f",
        a.unflatten(1, [groups, a.size(1) // groups]),  # B G C ...
        b.unflatten(0, [groups, b.size(0) // groups]),  # G OC IC ...
    ).flatten(1, 2)


def fft_conv(
    signal: Tensor,
    kernel: Tensor,
    bias: Tensor = None,
    stride: Union[int, Iterable[int]] = 1,
    padding: Union[int, Iterable[int]] = 0,
    dilation: Union[int, Iterable[int]] = 1,
    groups: int = 1,
    padding_mode: str = "constant",
    tradeoff: float = 0.5,
    blocksize: Union[int, Iterable[int]] = None,
) -> Tensor:
    """Performs N-d convolution of Tensors using a fast fourier transform, which
    is very fast for large kernel sizes. Also, optionally adds a bias Tensor after
    the convolution (in order ot mimic the PyTorch direct convolution).
    Args:
        signal: (Tensor) Input tensor to be convolved with the kernel.
        kernel: (Tensor) Convolution kernel.
        bias: (Tensor) Bias tensor to add to the output.
        stride: (Union[int, Iterable[int]) Stride size for computing output values.
        padding: (Union[int, Iterable[int]) Number of zero samples to pad the
            input on the last dimension.
        tradeoff: (float) Speed - Memory trade off
        blocksize: (Union[int, Iterable[int]) Manual tuning for debugging blocksize
    Returns:
        (Tensor) Convolved tensor
    """

    # Cast padding, stride & dilation to tuples.
    n = signal.ndim - 2
    padding_ = to_ntuple(padding, n=n)
    stride_ = to_ntuple(stride, n=n)
    dilation_ = to_ntuple(dilation, n=n)

    if any(x != 1 for x in dilation_):
        kernel_ = kernel.new_zeros(
            list(kernel.shape[:2])
            + [(k - 1) * d + 1 for k, d in zip(kernel.shape[2:], dilation_)]
        )
        kernel_[
            (slice(None), slice(None), *(slice(None, None, d) for d in dilation_))
        ] = kernel
        kernel = kernel_

    blocksize = (
        opt_blocksize(
            tuple(signal.shape[:2])
            + tuple(s + p * 2 for s, p in zip(signal.shape[2:], padding_)),
            kernel.shape,
            tradeoff,
        )
        if blocksize is None
        else to_ntuple(blocksize, n=n)
    )
    S = np.asarray(list(signal.shape[2:]), dtype=np.int64)
    K = np.asarray(list(kernel.shape[2:]), dtype=np.int64)
    B = np.asarray(list(blocksize), dtype=np.int64)
    ST = np.asarray(list(stride_), dtype=np.int64)
    PD = np.asarray(list(padding_), dtype=np.int64)
    BS = B + K - 1  # active block size
    GP = ((S + 2 * PD) - 1) // B * B + BS - S  # Global padding
    FFTss = (BS + 1) // 2 * 2  # fft signal size
    TRG = S + PD * 2 - K + 1  # target output shape

    signal_pad = F.pad(
        signal,
        [x for p, gp in zip(padding_, GP) for x in [p + gp, p]][::-1],
        padding_mode,
    )
    SP = np.asarray(signal_pad.shape[2:])
    unfold_signal = torch.nn.functional.unfold(
        signal_pad,
        kernel_size=BS,
        stride=B,
    ).unflatten(1, [signal.size(1)] + BS.tolist())

    signal_fr = rfftn(unfold_signal, FFTss.tolist(), dim=tuple(range(2, n + 2)))
    kernel_fr = rfftn(kernel, FFTss.tolist(), dim=tuple(range(2, n + 2)))
    output_fr = complex_matmul(signal_fr, kernel_fr.conj(), groups=groups)
    output = irfftn(output_fr, FFTss.tolist(), dim=tuple(range(2, n + 2)))
    output = output[[slice(None)] * 2 + [slice(None, b) for b in B]]
    output = torch.nn.functional.fold(
        output.flatten(1, n + 1),
        output_size=((SP - BS) // B + 1) * B,
        kernel_size=B,
        padding=0,
        stride=B,
    )
    output = output[[slice(None)] * 2 + [slice(None, t, s) for t, s in zip(TRG, ST)]]

    # Optionally, add a bias term before returning.
    if bias is not None:
        bias_shape = tuple([1, -1] + (signal.ndim - 2) * [1])
        output += bias.view(bias_shape)

    return output
