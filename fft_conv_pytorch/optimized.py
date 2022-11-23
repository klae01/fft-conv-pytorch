from collections import defaultdict
from functools import lru_cache
from typing import Iterable, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.fft import irfftn, rfftn

from .utils import to_ntuple


class prod:
    def __init__(self):
        self.items = defaultdict(lambda: 1)
        self.index = 0

    def reset(self):
        self.index = 0

    def put(self, x):
        self.items[self.index] = self.items[self.index] * x
        self.index += 1

    def get(self):
        value = self.items[self.index]
        self.index += 1
        return value


class accm(prod):
    def __init__(self):
        self.items = defaultdict(lambda: 0)
        self.index = 0

    def put(self, x):
        self.items[self.index] = self.items[self.index] + x
        self.index += 1


@lru_cache(1024)
def opt_blocksize(input_size, kernel_size, speed_impt=0.5):
    K = np.asarray(kernel_size[2:])
    S = np.asarray(input_size[2:])

    cost = prod()
    log_cost = accm()
    cost.put(np.prod(kernel_size[:2]))
    cost.put(np.prod(input_size[:2]))
    cost.put(input_size[0] * np.prod(kernel_size[:2]))
    cost.put(input_size[0] * kernel_size[0])

    cost.put(np.prod(kernel_size[:2]))
    cost.put(np.prod(input_size[:2]))
    cost.put(input_size[0] * kernel_size[0])
    cost.put(input_size[0] * kernel_size[0])

    for i, (s, k) in enumerate(zip(S, K)):
        shape = [1] * (len(input_size) - 2)
        shape[i] = -1
        B = np.arange(1, s + 1).reshape(*shape)

        BS = B + k - 1  # active block size
        GP = (s - k) // B * B + BS - s  # Global padding
        BC = (s + GP) // B
        log = np.ceil(np.log2(BS))
        kernel_cost = 2**log
        signal_cost = BC * 2**log

        cost.reset()
        log_cost.reset()

        log_cost.put(log)
        cost.put(kernel_cost)
        cost.put(signal_cost)
        cost.put(BC * BS)
        cost.put(signal_cost)

        cost.put(kernel_cost)
        cost.put(signal_cost)
        cost.put(BC * BS)
        cost.put(signal_cost)
    cost.reset()
    log_cost.reset()

    log_cost = log_cost.get()
    time_cost = [cost.get() for _ in range(4)]
    time_cost[0] *= log_cost
    time_cost[1] *= log_cost
    time_cost[3] *= log_cost
    time_cost = sum(time_cost)

    memo_cost = [cost.get() for _ in range(4)]
    memo_cost = sum(memo_cost)

    tradoff = time_cost ** (speed_impt) * memo_cost ** (1 - speed_impt)
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
    GP = (S + 2 * PD - K) // B * B + BS - S  # Global padding
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
        kernel_size=BS.tolist(),
        stride=B.tolist(),
    ).unflatten(1, [signal.size(1)] + BS.tolist())

    signal_fr = rfftn(unfold_signal, FFTss.tolist(), dim=tuple(range(2, n + 2)))
    kernel_fr = rfftn(kernel, FFTss.tolist(), dim=tuple(range(2, n + 2)))
    output_fr = complex_matmul(signal_fr, kernel_fr.conj(), groups=groups)
    output = irfftn(output_fr, FFTss.tolist(), dim=tuple(range(2, n + 2)))
    output = output[[slice(None)] * 2 + [slice(None, b) for b in B]]
    output = torch.nn.functional.fold(
        output.flatten(1, n + 1),
        output_size=(((SP - BS) // B + 1) * B).tolist(),
        kernel_size=B.tolist(),
        padding=0,
        stride=B.tolist(),
    )
    output = output[[slice(None)] * 2 + [slice(None, t, s) for t, s in zip(TRG, ST)]]

    # Optionally, add a bias term before returning.
    if bias is not None:
        bias_shape = tuple([1, -1] + (signal.ndim - 2) * [1])
        output += bias.view(bias_shape)

    return output
