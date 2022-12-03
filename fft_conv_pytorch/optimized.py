from collections import defaultdict
from functools import lru_cache
from typing import Iterable, Tuple, Union

import numba
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.fft import irfftn, rfftn

from .functional import complex_matmul
from .utils import to_ntuple


@numba.njit()
def get_maximum_divider(n, end):
    assert 1 <= n
    div = np.arange(1, np.sqrt(n) + 2, dtype=np.int64)
    div = div[n % div == 0]
    if n // div[-1] <= end:
        value = n // div[np.searchsorted(div, (n + end - 1) // end)]
    else:
        value = div[np.searchsorted(div, end, "right") - 1]
    assert 1 <= value <= end
    return value


@numba.njit()
def _candidates(s, k, BS, RET):
    index = 0
    for blocksize in BS:
        depad_simplity, depad_index = [s * 2] * 4, np.int64(-1)
        enpad_simplity, enpad_index = [s * 2] * 4, np.int64(-1)
        max_ssbc = (s + blocksize - k * 2 + 1) // (blocksize - k + 1)
        min_ssbc = max_ssbc if (blocksize - k + 1) * max_ssbc + k - 1 == s else 1
        for subsection_blockcount in range(min_ssbc, max_ssbc + 1):
            section_validsize = (blocksize - k + 1) * subsection_blockcount
            section_blocksize = section_validsize + k - 1
            if section_blocksize < s:
                # zero padding > minimum ops greedy
                jump = get_maximum_divider(s - section_blocksize, section_validsize)
                subsections = (s - section_blocksize) // jump + 1
                blockcount = subsections * subsection_blockcount
                padding = jump * (subsections - 1) + section_blocksize - s
                assert padding == 0
                simplity = [
                    blockcount,
                    padding,
                    section_blocksize > s,
                    -subsection_blockcount,
                ]
                if depad_simplity > simplity:
                    if depad_index == -1:
                        depad_index = index
                        index += 1
                    depad_simplity = simplity
                    RET[depad_index] = [
                        subsections,
                        jump,
                        subsection_blockcount,
                        blocksize,
                        padding,
                    ]

            # minimum ops > minimum padding greedy
            subsections = (s - k) // section_validsize + 1
            assert section_validsize * (subsections - 1) + section_blocksize >= s
            blockcount = subsections * subsection_blockcount
            jump = section_validsize
            padding = jump * (subsections - 1) + section_blocksize - s
            if subsections > 1:
                jump -= padding // (subsections - 1)
            padding = jump * (subsections - 1) + section_blocksize - s
            simplity = [
                blockcount,
                padding,
                section_blocksize > s,
                -subsection_blockcount,
            ]
            if enpad_simplity > simplity:
                if enpad_index == -1:
                    enpad_index = index
                    index += 1
                enpad_simplity = simplity
                RET[enpad_index] = [
                    subsections,
                    jump,
                    subsection_blockcount,
                    blocksize,
                    padding,
                ]
        if depad_index != -1 and enpad_index != -1:
            if (RET[depad_index] == RET[enpad_index]).all():
                index -= 1
    return index


def candidates(s, k):
    # subsections / jump / subsection_blockcount / blocksize / padding
    BS = [1]
    for p in [2, 3, 5, 7]:
        BS.append(p)
        while BS[-1] < s:
            BS.append(BS[-1] * p)
    BS = np.sort(np.array(BS, np.int64))
    BS = BS[BS >= k]
    RET = np.empty([len(BS) * 2, 5], dtype=np.int64)
    return RET[: _candidates(s, k, BS, RET)]


@lru_cache(1024)
def opt_control(
    input_size,
    kernel_size,
    speed_impt=0.5,
    real_input=True,
    padpenalty=True,
    backward=True,
):
    K = np.asarray(kernel_size[2:])
    S = np.asarray(input_size[2:])
    n = len(K)

    size_candidate = list(map(candidates, S, K))
    ops_size = list(map(len, size_candidate))
    c_info = np.ones(ops_size, dtype=np.int64)  # block count
    e_info = np.zeros(ops_size, dtype=np.int64)  # log of block size
    s_info = np.ones(ops_size, dtype=np.int64)  # block size
    p_info = np.ones(ops_size, dtype=np.int64)  # plane
    d_info = np.zeros(ops_size + [n], dtype=np.int64)  # padding
    b_info = np.zeros(ops_size + [n], dtype=np.int64)  # block size

    for i, SCD in enumerate(size_candidate):
        shape = [1] * n
        shape[i] = -1
        BC1, _, BC2, BS, GP = map(lambda x: x.reshape(shape), SCD.T)
        BC = BC1 * BC2
        c_info *= BC
        e_info += np.ceil(np.log2(BS)).astype(e_info.dtype)
        s_info *= BS
        p_info *= [BS, (BS // 2 + 1) * 2][i == n - 1 and real_input]
        d_info[..., i] = GP
        b_info[..., i] = BS

    # Assume cuFFT is as follows:
    # time complexity = n ⌈log2 n⌉
    # memory complexity = 2 ^ ⌈log2 n⌉
    sig_pad = ((d_info > 0).any(-1) | np.bool_(not padpenalty)).astype(d_info.dtype)
    krn_pad = (b_info != K).any(-1).astype(d_info.dtype)
    sig_cnt = (c_info != 1).astype(d_info.dtype)

    time_cost = np.prod(input_size[:2]) * (S + d_info).prod(-1) * sig_pad  # signal pad
    time_cost += np.prod(kernel_size[:2]) * b_info.prod(-1) * krn_pad  # kernel pad
    time_cost += np.prod(input_size[:2]) * c_info * s_info * sig_cnt  # sig contiguous
    time_cost += np.prod(kernel_size[:2]) * s_info * np.maximum(1, e_info)
    time_cost += np.prod(input_size[:2]) * c_info * s_info * np.maximum(1, e_info)
    time_cost += input_size[0] * np.prod(kernel_size[:2]) * c_info * p_info
    time_cost += (
        input_size[0] * kernel_size[0] * c_info * s_info * np.maximum(1, e_info)
    )

    def update(peak, curr, band=None):
        nonlocal memo_curr, memo_band, memo_peak
        np.maximum(memo_peak, memo_curr + peak, out=memo_peak)
        memo_band += [band, peak][band is None]
        memo_curr += curr

    memo_curr = np.prod(input_size[:2]) * (S + d_info).prod(-1) * sig_pad  # signal pad
    memo_curr += np.prod(kernel_size[:2]) * s_info * krn_pad  # kernal pad
    memo_curr += np.prod(input_size[:2]) * c_info * s_info * sig_cnt  # sig contiguous
    memo_band = memo_curr.copy()
    memo_peak = memo_curr.copy()
    update(
        np.prod(kernel_size[:2]) * s_info * 2,
        np.prod(kernel_size[:2]) * p_info,
        # np.prod(kernel_size[:2]) * s_info * e_info,
    )
    update(
        np.prod(input_size[:2]) * c_info * s_info * 2,
        np.prod(input_size[:2]) * c_info * p_info,
        # np.prod(input_size[:2]) * c_info * s_info * e_info,
    )
    sig_tensor = np.prod(input_size[:2]) * c_info * p_info
    krn_tensor = np.prod(kernel_size[:2]) * p_info
    out_tensor = input_size[0] * kernel_size[0] * c_info * p_info
    grad_save = sig_tensor + krn_tensor if backward else 0
    update(
        np.maximum(sig_tensor, krn_tensor) + out_tensor + grad_save,
        out_tensor + grad_save,
        input_size[0] * np.prod(kernel_size[:2]) * c_info * p_info
        + sig_tensor
        + krn_tensor
        + out_tensor,
    )
    update(
        input_size[0] * kernel_size[0] * c_info * s_info * 2,
        input_size[0] * kernel_size[0] * c_info * s_info,
        # input_size[0] * kernel_size[0] * c_info * s_info * e_info,
    )
    memo_cost = memo_peak.astype(np.float64)
    time_cost = (time_cost + memo_band * 4).astype(np.float64)

    tradoff = time_cost ** (speed_impt) * memo_cost ** (1 - speed_impt)
    index = (tradoff == np.min(tradoff)).nonzero()
    index = sorted(zip(*index))[0]
    return np.asarray(list(map(lambda x, y: x[y], size_candidate, index)))


# @numba.njit()
# def Dblock_indexing(section_element_count, valid_blocksize, stride, result):
#     rindex = 0
#     pindex = 0
#     for i, SEC in enumerate(section_element_count):
#         k_start = (stride - rindex % stride) % stride
#         ops = min(result.shape[1] - pindex, (SEC - k_start) // stride)
#         end = pindex + ops
#         result[0, pindex:end] = i
#         np.divmod(
#             np.arange(k_start, ops * stride + k_start, stride),
#             valid_blocksize,
#             result[1, pindex:end],
#             result[2, pindex:end],
#         )
#         pindex = end
#         rindex += SEC


@numba.njit()
def Dblock_indexing(section_case, valid_blocksize, stride, result):
    rindex = 0
    pindex = 0
    for i, SC in enumerate(section_case):
        for j in range(SC // valid_blocksize + 1):
            sub_block_ops = min(valid_blocksize, SC - j * valid_blocksize)
            k_start = (stride - rindex % stride) % stride
            sub_ops = max(0, (sub_block_ops - k_start + stride - 1) // stride)
            sub_ops = min(sub_ops, result.shape[1] - pindex)
            end = pindex + sub_ops
            result[0, pindex:end] = i
            result[1, pindex:end] = j
            result[2, pindex:end] = np.arange(
                k_start, sub_ops * stride + k_start, stride
            )
            pindex += sub_ops
            rindex += sub_block_ops
            if result.shape[1] == pindex:
                return


@torch.no_grad()
def Dblock(tensor: Tensor, K, TRG, ST, SSC, JMP, SBC, BSZ):
    # B C H W -> B C HSSC HSBC WSSC WSBC HBSZ WBSZ
    def reorder(X):
        nonlocal n
        core = [4 + i * 3 for i in range(n)]
        return [X[i] for i in range(len(X)) if i not in core] + [X[i] for i in core]

    n = tensor.ndim - 2
    S = list(tensor.shape)
    T = list(tensor.stride())
    size, S = S[:2], S[2:]
    stride, T = T[:2], T[2:]
    merge_index = [slice(None)] * 2
    for i, t in enumerate(T):
        valid_blocksize = BSZ[i] - K[i] + 1
        size.append(SSC[i])
        size.append(SBC[i])
        size.append(BSZ[i])
        stride.append(t * JMP[i])
        stride.append(t * valid_blocksize)
        stride.append(t)
        section_element_count = [JMP[i]] * (SSC[i] - 1) + [valid_blocksize * SBC[i]]
        npresult = np.empty([3, (TRG[i] + ST[i] - 1) // ST[i]], dtype=np.int64)
        ptresult = tensor.new_empty(npresult.shape, dtype=torch.long)
        Dblock_indexing(
            np.array(section_element_count), valid_blocksize, ST[i], npresult
        )
        ptresult.copy_(torch.from_numpy(npresult), non_blocking=True)
        index_shape = [[1, -1][i == idx] for idx in range(n)]
        merge_index += ptresult.reshape(3, *index_shape)
    size, stride, merge_index = map(reorder, [size, stride, merge_index])
    return tensor.as_strided(size, stride), merge_index


def Mblock(tensor: Tensor, merge_index):
    return tensor[merge_index]


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

    control = opt_control(
        tuple(signal.shape[:2])
        + tuple(s + p * 2 for s, p in zip(signal.shape[2:], padding_)),
        kernel.shape,
        tradeoff,
        real_input=not any(map(torch.is_complex, [signal, kernel])),
        padpenalty=all(x == 0 for x in padding_),
        backward=any(M.requires_grad for M in [signal, kernel]),
    )
    SSC, JMP, SBC, BSZ, PAD = control.T
    S = np.asarray(list(signal.shape[2:]), dtype=np.int64)
    K = np.asarray(list(kernel.shape[2:]), dtype=np.int64)
    ST = np.asarray(list(stride_), dtype=np.int64)
    PD = np.asarray(list(padding_), dtype=np.int64)
    TRG = S + PD * 2 - K + 1  # target output shape

    if PD.any() or PAD.any():
        signal_padding = [x for p, gp in zip(PD, PAD) for x in [p + gp, p]][::-1]
        signal = F.pad(signal, signal_padding, mode=padding_mode)

    signal, merge_index = Dblock(signal, K, TRG, ST, SSC, JMP, SBC, BSZ)
    kernel = kernel[[slice(None)] * 2 + [None] * 2 * n + [slice(None)] * n]
    
    signal_fr = rfftn(signal, BSZ.tolist(), dim=tuple(range(-n, 0)))
    kernel_fr = rfftn(kernel, BSZ.tolist(), dim=tuple(range(-n, 0)))
    output_fr = complex_matmul(signal_fr, kernel_fr.conj(), groups=groups)
    output = irfftn(output_fr, BSZ.tolist(), dim=tuple(range(-n, 0)))
    output = Mblock(output, merge_index)

    # Optionally, add a bias term before returning.
    if bias is not None:
        bias_shape = tuple([1, -1] + [1] * n)
        output = bias.view(bias_shape) + output

    return output
