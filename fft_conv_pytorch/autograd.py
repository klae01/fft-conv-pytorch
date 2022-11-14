from typing import Iterable, Tuple, Union
from torch import Tensor, nn

conv_AutoGrad = ...
conv_transpose_AutoGrad = ...

import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable
from torch.fft import irfftn, rfftn

from .functional import fft_conv, fft_conv_transpose


class conv_AutoGrad(torch.autograd.Function):
    @staticmethod
    def complex_matmul(a: Tensor, b: Tensor, groups: int = 1) -> Tensor:
        return torch.einsum(
            "bgi...,bgo...->goi...",
            a.unflatten(1, [groups, a.size(1) // groups]),  # B G C ...
            b.unflatten(1, [groups, b.size(1) // groups]),  # G OC IC ...
        ).flatten(0, 1)

    @staticmethod
    def crop_or_pad(input: Tensor, size):
        print(input.shape, size)
        assert input.ndim == len(size)
        assert list(input.shape[:2]) == list(size[:2])
        slicing = [slice(None, min(*x)) for x in zip(input.shape, size)]
        if all(x >= y for x, y in zip(input.shape, size)):
            return input[slicing]
        output = input.new_zeros(size)
        output[slicing] = input[slicing]
        return output

    @staticmethod
    def variable_encode(*args):
        def _sub(x):
            if isinstance(x, Tensor):
                return True, x
            else:
                return False, torch.tensor(x, requires_grad=False)

        is_tensor, result = zip(*map(_sub, args))
        return torch.tensor(list(is_tensor)), *result

    @staticmethod
    def variable_decode(is_tensor, *variable):
        def _sub(is_tensor, x):
            if is_tensor:
                return x
            else:
                return x.numpy().tolist()

        return map(_sub, is_tensor.numpy(), variable)

    @staticmethod
    def forward(ctx, signal, kernel, bias, stride, dilation, groups):
        ctx.save_for_backward(
            *conv_AutoGrad.variable_encode(
                signal,
                kernel,
                signal.shape,
                kernel.shape,
                bias is not None,
                stride,
                dilation,
                groups,
            )
        )
        return fft_conv(signal, kernel, bias, stride, 0, dilation, groups)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out: Tensor):
        (
            signal,
            kernel,
            signal_shape,
            kernel_shape,
            use_bias,
            stride,
            dilation,
            groups,
        ) = conv_AutoGrad.variable_decode(*ctx.saved_tensors)
        grad_kernel = fft_conv(
            signal.unflatten(1, [groups, -1]).transpose(0, 2).flatten(1, 2),
            grad_out.transpose(0, 1),
            None,
            dilation,
            0,
            stride,
            groups,
        ).transpose(0, 1)
        grad_input = fft_conv_transpose(
            grad_out, kernel, None, stride, 0, 0, dilation, groups
        )
        grad_bias = grad_out.sum([0, *range(2, grad_out.ndim)]) if use_bias else None

        return (
            conv_AutoGrad.crop_or_pad(grad_input, signal_shape),
            conv_AutoGrad.crop_or_pad(grad_kernel, kernel_shape),
            grad_bias,
            None,
            None,
            None,
        )
