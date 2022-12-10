import torch
from torch import Tensor

from .lib import cpu, cuda


def RaiseNotImplementedError(str):
    def _sub(*args):
        raise NotImplementedError(str)

    return _sub


def get_device_type(tensor: Tensor):
    if tensor.device == torch.device("cpu"):
        return "CPU"
    elif torch.cuda.is_available() and torch.version.cuda and tensor.is_cuda:
        return "CUDA"
    elif torch.cuda.is_available() and torch.version.hip and tensor.is_cuda:
        return "HIP"
    else:
        raise NotImplementedError("Unable to decide Tensor device type")


class FastPlaneDot(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a: Tensor, b: Tensor):
        assert a.is_contiguous(memory_format=torch.contiguous_format)
        assert b.is_contiguous(memory_format=torch.contiguous_format)
        assert a.dtype == b.dtype
        assert a.device == b.device
        assert a.ndim == 5 and b.ndim == 4
        _, ag, ai, _, al = a.shape
        bg, _, bi, bl = b.shape
        assert ag == bg
        assert ai == bi
        assert al == bl
        ctx.save_for_backward(a, b)
        result = {
            "CPU": cpu.PlaneDot_forward,
            "CUDA": cuda.PlaneDot_forward,
            "HIP": RaiseNotImplementedError("Device type: HIP"),
        }[get_device_type(a)](a, b)
        result.requires_grad = a.requires_grad or b.requires_grad
        return result

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_A = a.new_empty(a.shape)
        grad_B = b.new_empty(b.shape)
        {
            "CPU": cpu.PlaneDot_backprop_A,
            "CUDA": cuda.PlaneDot_backprop_A,
            "HIP": RaiseNotImplementedError("Device type: HIP"),
        }[get_device_type(a)](grad_A, b, grad_output)
        {
            "CUDA": cuda.PlaneDot_backprop_B,
            "CPU": cpu.PlaneDot_backprop_B,
            "HIP": RaiseNotImplementedError("Device type: HIP"),
        }[get_device_type(a)](a, grad_B, grad_output)
        return grad_A, grad_B
