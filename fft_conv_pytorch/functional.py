from typing import Iterable, Tuple, Union

import torch
import torch.nn.functional as f
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

    # internal dilation offsets
    offset = torch.zeros(1, 1, *dilation_, device=signal.device, dtype=signal.dtype)
    offset[(slice(None), slice(None), *((0,) * n))] = 1.0

    # correct the kernel by cutting off unwanted dilation trailing zeros
    cutoff = tuple(slice(None, -d + 1 if d != 1 else None) for d in dilation_)

    # pad the kernel internally according to the dilation parameters
    kernel = torch.kron(kernel, offset)[(slice(None), slice(None)) + cutoff]

    # Pad the input signal & kernel tensors
    signal_padding = [p for p in padding_[::-1] for _ in range(2)]
    signal = f.pad(signal, signal_padding, mode=padding_mode)

    # Because PyTorch computes a *one-sided* FFT, we need the final dimension to
    # have *even* length.  Just pad with one more zero if the final dimension is odd.
    interm_shape = [(s + 1) // 2 * 2 for s in signal.shape[2:]]

    # Perform fourier convolution -- FFT, matrix multiply, then IFFT
    signal_fr = rfftn(signal, interm_shape, dim=tuple(range(2, signal.ndim)))
    kernel_fr = rfftn(kernel, interm_shape, dim=tuple(range(2, signal.ndim)))

    output_fr = complex_matmul(signal_fr, kernel_fr.conj(), groups=groups)
    output = irfftn(output_fr, dim=tuple(range(2, signal.ndim)))

    # Remove extra padded values
    crop_slices = [slice(0, output.size(0)), slice(0, output.size(1))] + [
        slice(0, (signal.size(i) - kernel.size(i) + 1), stride_[i - 2])
        for i in range(2, signal.ndim)
    ]
    output = output[crop_slices].contiguous()

    # Optionally, add a bias term before returning.
    if bias is not None:
        bias_shape = tuple([1, -1] + (signal.ndim - 2) * [1])
        output += bias.view(bias_shape)

    return output
