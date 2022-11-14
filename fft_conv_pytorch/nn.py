from torch import Tensor, nn
import torch.nn.functional as F

from .autograd import conv_AutoGrad, conv_transpose_AutoGrad
from .functional import fft_conv, fft_conv_transpose
from .utils import to_ntuple


class _FFTConvForward(nn.Module):
    __low_memory = False

    def set_memory_optimize(self, enable: bool = True):
        self.__low_memory = enable

    def _AutoGrad(self, **kwargs):
        if self.__low_memory:
            padding = kwargs.pop("padding")
            padding_mode = kwargs.pop("padding_mode")
            if any(x != 0 for x in padding):
                signal = kwargs.pop("signal")
                kwargs["signal"] = F.pad(signal, padding, mode=padding_mode)
            return conv_AutoGrad.apply(**kwargs)

    def forward(self, signal: Tensor):
        assert signal.ndim == self.weight.ndim
        padding_mode = [self.padding_mode, "constant"][self.padding_mode == "zeros"]
        if self.__low_memory:
            if any(x != 0 for x in self.padding):
                padding = [p for p in self.padding[::-1] for _ in range(2)]
                signal = F.pad(signal, padding, mode=padding_mode)
            return conv_AutoGrad.apply(
                signal, self.weight, self.bias, self.stride, self.dilation, self.groups
            )
        else:
            return fft_conv(
                signal,
                self.weight,
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
                padding_mode,
            )


class _FFTConvTransposeForward(nn.Module):
    __low_memory = False

    def set_memory_optimize(self, enable: bool = True):
        self.__low_memory = enable

    def _AutoGrad(self, **kwargs):
        raise NotImplementedError

    def forward(self, signal: Tensor):
        assert signal.ndim == self.weight.ndim
        return (self._AutoGrad if self.__low_memory else fft_conv_transpose)(
            signal,
            self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            dilation=self.dilation,
            groups=self.groups,
        )


class FFTConv1d(_FFTConvForward, nn.Conv1d):
    ...


class FFTConv2d(_FFTConvForward, nn.Conv2d):
    ...


class FFTConv3d(_FFTConvForward, nn.Conv3d):
    ...


class FFTConvTranspose1d(_FFTConvTransposeForward, nn.ConvTranspose1d):
    ...


class FFTConvTranspose2d(_FFTConvTransposeForward, nn.ConvTranspose2d):
    ...


class FFTConvTranspose3d(_FFTConvTransposeForward, nn.ConvTranspose3d):
    ...
