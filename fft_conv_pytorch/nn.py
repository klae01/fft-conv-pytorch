from functools import partial
from typing import Iterable, Tuple, Union

import torch
from torch import Tensor, nn
from torch.nn.modules.conv import _ConvNd

from .functional import fft_conv
from .utils import to_ntuple


class _FFTConvNd(_ConvNd):
    """Base class for PyTorch FFT convolution layers."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Iterable[int]],
        stride: Union[int, Iterable[int]] = 1,
        padding: Union[int, str, Iterable[int]] = 0,
        dilation: Union[int, Iterable[int]] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
        ndim: int = 1,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        kernel_size_ = to_ntuple(kernel_size, ndim)
        stride_ = to_ntuple(stride, ndim)
        padding_ = padding if isinstance(padding, str) else to_ntuple(padding, ndim)
        dilation_ = to_ntuple(dilation, ndim)
        super(_FFTConvNd, self).__init__(
            in_channels,
            out_channels,
            kernel_size_,
            stride_,
            padding_,
            dilation_,
            False,
            to_ntuple(0, ndim),
            groups,
            bias,
            padding_mode,
            **factory_kwargs
        )

    def forward(self, signal: Tensor):
        assert signal.ndim == self.weight.ndim
        padding_mode = [self.padding_mode, "constant"][self.padding_mode == "zeros"]
        return fft_conv(
            signal,
            self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            padding_mode=padding_mode,
        )


FFTConv1d = partial(_FFTConvNd, ndim=1)
FFTConv2d = partial(_FFTConvNd, ndim=2)
FFTConv3d = partial(_FFTConvNd, ndim=3)
