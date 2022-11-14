from typing import Iterable, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.fft import irfftn, rfftn

from .utils import to_ntuple


def complex_matmul(a: Tensor, b: Tensor, groups: int = 1) -> Tensor:
    return torch.einsum(
        "bgi...,goi...->bgo...",
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

    # Pad the input signal & kernel tensors
    if any(x != 0 for x in padding_):
        signal_padding = [p for p in padding_[::-1] for _ in range(2)]
        signal = F.pad(signal, signal_padding, mode=padding_mode)

    # Because PyTorch computes a *one-sided* FFT, we need the final dimension to
    # have *even* length.  Just pad with one more zero if the final dimension is odd.
    interm_shape = [(s + 1) // 2 * 2 for s in signal.shape[2:]]

    # Perform fourier convolution -- FFT, matrix multiply, then IFFT
    signal_fr = rfftn(signal, interm_shape, dim=tuple(range(2, signal.ndim)))
    kernel_fr = rfftn(kernel, interm_shape, dim=tuple(range(2, signal.ndim)))

    output_fr = complex_matmul(signal_fr, kernel_fr.conj(), groups=groups)
    output = irfftn(output_fr, dim=tuple(range(2, signal.ndim)))

    # Remove extra padded values
    output = output[
        [slice(0, output.size(0)), slice(0, output.size(1))]
        + [
            slice(0, (signal.size(i) - kernel.size(i) + 1), stride_[i - 2])
            for i in range(2, signal.ndim)
        ]
    ]

    # Optionally, add a bias term before returning.
    if bias is not None:
        bias_shape = tuple([1, -1] + (signal.ndim - 2) * [1])
        output += bias.view(bias_shape)

    return output


def fft_conv_transpose(
    signal: Tensor,
    kernel: Tensor,
    bias: Tensor = None,
    stride: Union[int, Iterable[int]] = 1,
    padding: Union[int, Iterable[int]] = 0,
    output_padding: Union[int, Iterable[int]] = 0,
    dilation: Union[int, Iterable[int]] = 1,
    groups: int = 1,
) -> Tensor:
    # Cast padding, stride & dilation to tuples.
    n = signal.ndim - 2
    padding_ = to_ntuple(padding, n=n)
    output_padding_ = to_ntuple(output_padding, n=n)
    stride_ = to_ntuple(stride, n=n)
    dilation_ = to_ntuple(dilation, n=n)

    kernel = (
        kernel.flip(*range(2, signal.ndim))
        .unflatten(0, [groups, kernel.size(0) // groups])
        .transpose_(1, 2)
        .flatten(0, 1)
    )
    if any(x != 1 for x in dilation_):
        kernel_ = kernel.new_zeros(
            list(kernel.shape[:2])
            + [(k - 1) * d + 1 for k, d in zip(kernel.shape[2:], dilation_)]
        )
        kernel_[
            (slice(None), slice(None), *(slice(None, None, d) for d in dilation_))
        ] = kernel
    else:
        kernel_ = kernel

    signal_ = signal.new_zeros(
        list(signal.shape[:2])
        + [
            (s - 1) * t + 1 + (k - 1)
            for s, k, t in zip(signal.shape[2:], kernel_.shape[2:], stride_)
        ]
    )
    signal_[
        (
            slice(None),
            slice(None),
            *(slice(k - 1, None, t) for k, t in zip(kernel_.shape[2:], stride_)),
        )
    ] = signal

    # Because PyTorch computes a *one-sided* FFT, we need the final dimension to
    # have *even* length.  Just pad with one more zero if the final dimension is odd.
    interm_shape = [(s + k) // 2 * 2 for s, k in zip(signal_.shape, kernel_.shape)][2:]
    output_shape = [
        (s - 1) * t - 2 * p + d * (k - 1) + o + 1
        for s, k, t, p, d, o in zip(
            signal.shape[2:],
            kernel.shape[2:],
            stride_,
            padding_,
            dilation_,
            output_padding_,
        )
    ]

    # Perform fourier convolution -- FFT, matrix multiply, then IFFT
    signal_fr = rfftn(signal_, interm_shape, dim=tuple(range(2, signal.ndim)))
    kernel_fr = rfftn(kernel_, interm_shape, dim=tuple(range(2, signal.ndim)))

    output_fr = complex_matmul(signal_fr, kernel_fr.conj(), groups=groups)
    output = irfftn(output_fr, dim=tuple(range(2, signal.ndim)))
    output = output[
        (
            slice(None),
            slice(None),
            *(slice(p, s + p) for s, p in zip(output_shape, padding_)),
        )
    ]

    # Optionally, add a bias term before returning.
    if bias is not None:
        bias_shape = tuple([1, -1] + (signal.ndim - 2) * [1])
        output += bias.view(bias_shape)

    return output
