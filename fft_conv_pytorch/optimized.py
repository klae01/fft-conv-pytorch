from collections import defaultdict
from functools import lru_cache
from typing import Iterable, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.fft import irfftn, rfftn

from .functional import complex_matmul
from .utils import to_ntuple


def blocksize_candidate(start, end):
    def _sub(x):
        for p in [2, 3, 5, 7]:
            while x % p == 0:
                x //= p
        return x == 1

    return [v for v in range(start, end + 1) if _sub(v)]


@lru_cache(1024)
def opt_blocksize(input_size, kernel_size, speed_impt=0.5):
    K = np.asarray(kernel_size[2:])
    S = np.asarray(input_size[2:])
    n = len(K)

    size_candidate = list(map(blocksize_candidate, K, S * 7))
    ops_size = list(map(len, size_candidate))
    c_info = np.ones(ops_size, dtype=np.float64)
    e_info = np.zeros(ops_size, dtype=np.float64)
    s_info = np.ones(ops_size, dtype=np.float64)

    for i, (k, s, bs) in enumerate(zip(K, S, size_candidate)):
        shape = [1] * n
        shape[i] = -1
        BS = np.asarray(bs).reshape(shape)
        B = BS - k + 1  # valid size
        GP = (s - k) // B * B + BS - s  # Global padding
        BC = (s - k + 1 + GP) // B
        c_info *= BC
        e_info += np.ceil(np.log2(BS))
        s_info *= BS

    # Assume cuFFT is as follows:
    # time complexity = n ⌈log2 n⌉
    # memory complexity = 2 ^ ⌈log2 n⌉
    time_cost = np.prod(kernel_size[:2]) * s_info * e_info
    time_cost += np.prod(input_size[:2]) * c_info * s_info * e_info
    time_cost += input_size[0] * np.prod(kernel_size[:2]) * c_info * s_info
    time_cost += input_size[0] * kernel_size[0] * c_info * s_info * e_info

    memo_cost = np.prod(kernel_size[:2]) * 2**e_info
    memo_cost += np.prod(input_size[:2]) * c_info * 2**e_info
    memo_cost += input_size[0] * kernel_size[0] * c_info * s_info
    memo_cost += input_size[0] * kernel_size[0] * c_info * 2**e_info

    tradoff = time_cost ** (speed_impt) * memo_cost ** (1 - speed_impt)
    index = (tradoff == np.min(tradoff)).nonzero()
    index = sorted(zip(*index))[0]
    return list(map(lambda x, y, z: x[y] - z + 1, size_candidate, index, K))


def Dblock(tensor: Tensor, blocksize: Iterable[int], active_blocksize: Iterable[int]):
    n = tensor.ndim - 2
    S = list(tensor.shape)
    B = list(blocksize)
    A = list(active_blocksize)
    T = list(tensor.stride())
    size = S[:2]
    stride = T[:2]
    for s, b, a, t in zip(S[2:], B, A, T[2:]):
        size.append((s - a) // b + 1)
        stride.append(t * b)
        size.append(a)
        stride.append(t)
    reorder, wait = [], []
    for i in range(n * 2 + 2):
        if i < 2 or i % 2 == 0:
            reorder.append(i)
        else:
            wait.append(i)
    reorder += wait
    return tensor.as_strided(size, stride, tensor.storage_offset()).permute(reorder)


def Mblock(tensor: Tensor):
    n = (tensor.ndim - 2) // 2
    q1 = list(range(n + 2))
    q2 = list(range(n + 2, n * 2 + 2))
    reorder = []
    while q1 or q2:
        if q2:
            reorder.append(q2.pop())
        if q1:
            reorder.append(q1.pop())
    tensor = tensor.permute(reorder[::-1])
    S = list(tensor.shape)
    for i in range(2, 2 + n):
        S[i] = S[i] * S[i + 1]
        S.pop(i + 1)
    return tensor.reshape(S)


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
        kernel_[[slice(None)] * 2 + [slice(None, None, d) for d in dilation_]] = kernel
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
    FFTss = BS  # fft signal size
    TRG = S + PD * 2 - K + 1  # target output shape

    signal_pad = F.pad(
        signal,
        [x for p, gp in zip(padding_, GP) for x in [p + gp, p]][::-1],
        padding_mode,
    )
    block_signal = Dblock(signal_pad, B, BS)
    for _ in range(n):
        kernel = kernel.unsqueeze(2)

    signal_fr = rfftn(block_signal, FFTss.tolist(), dim=tuple(range(-n, 0)))
    kernel_fr = rfftn(kernel, FFTss.tolist(), dim=tuple(range(-n, 0)))
    output_fr = complex_matmul(signal_fr, kernel_fr.conj(), groups=groups)
    output = irfftn(output_fr, FFTss.tolist(), dim=tuple(range(-n, 0)))
    output = output[[slice(None)] * (2 + n) + [slice(None, b) for b in B]]
    output = Mblock(output)
    output = output[[slice(None)] * 2 + [slice(None, t, s) for t, s in zip(TRG, ST)]]

    # Optionally, add a bias term before returning.
    if bias is not None:
        bias_shape = tuple([1, -1] + (signal.ndim - 2) * [1])
        output += bias.view(bias_shape)

    return output
