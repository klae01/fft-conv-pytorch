from torch import Tensor, nn

from .functional import fft_conv  # , fft_conv_transpose
from ..utils import to_ntuple


class _FFTConvForward(nn.Module):
    """Base class for PyTorch FFT convolution layers."""

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


# class _FFTConvTransposeForward(nn.Module):
#     """Base class for PyTorch FFT convolution layers."""

#     def forward(self, signal: Tensor):
#         assert signal.ndim == self.weight.ndim
#         return fft_conv_transpose(
#             signal,
#             self.weight,
#             bias=self.bias,
#             stride=self.stride,
#             padding=self.padding,
#             output_padding=self.output_padding,
#             dilation=self.dilation,
#             groups=self.groups,
#         )


class FFTConv1d(_FFTConvForward, nn.Conv1d):
    ...


class FFTConv2d(_FFTConvForward, nn.Conv2d):
    ...


class FFTConv3d(_FFTConvForward, nn.Conv3d):
    ...


# class FFTConvTranspose1d(_FFTConvTransposeForward, nn.ConvTranspose1d):
#     ...


# class FFTConvTranspose2d(_FFTConvTransposeForward, nn.ConvTranspose2d):
#     ...


# class FFTConvTranspose3d(_FFTConvTransposeForward, nn.ConvTranspose3d):
#     ...
