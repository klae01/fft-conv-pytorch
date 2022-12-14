import torch
import numpy as np
import numba
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


@numba.njit()
def case_generate(N, limit):
    SQ = int(np.ceil(np.sqrt(N))) + 1
    while SQ * SQ > N:
        SQ -= 1
    for x in range(1, min(limit + 1, SQ)):
        yield x
    for x in range((N + limit - 1) // limit, SQ + 1):
        yield N // x


@numba.njit()
def case_count(depth, max_ops, usage, ARR, apply):
    if depth == len(ARR):
        return 1
    sum = 0
    remain = max_ops // usage[apply[depth]].sum()
    if remain:
        tmp_usage = usage.copy()
        for i in case_generate(ARR[depth], remain):
            tmp_usage[...] = usage
            tmp_usage[apply[depth]] *= i
            sum += case_count(depth + 1, max_ops, tmp_usage, ARR, apply)
    return sum


@numba.njit()
def case_memo(depth, max_ops, usage, ARR, apply, index, stack_result, result):
    if depth == len(ARR):
        result[index] = stack_result
        return index + 1
    remain = max_ops // usage[apply[depth]].sum()
    if remain:
        tmp_usage = usage.copy()
        for i in case_generate(ARR[depth], remain):
            stack_result[depth] = i
            tmp_usage[...] = usage
            tmp_usage[apply[depth]] *= i
            index = case_memo(
                depth + 1, max_ops, tmp_usage, ARR, apply, index, stack_result, result
            )
    return index


@torch.no_grad()
def memory_access_count(DIM: np.ndarray, DIV: np.ndarray, stride: list, chunksize: int):
    # bottom up, memory shift 0 ~ chunksize, all case cache
    # ignore out of range
    # TODO: P1, consider offset
    u_DIV, indices = np.unique(DIV, axis=0, return_inverse=True)

    result = torch.zeros([u_DIV.shape[0]], dtype=torch.float32)
    cpu.AccessCost(
        torch.tensor([DIM.shape[0]]).type(torch.int64),
        torch.tensor(chunksize).type(torch.int64),
        torch.from_numpy(DIM.astype(np.int64)),
        torch.tensor(stride).type(torch.int64),
        torch.from_numpy(u_DIV.astype(np.int64)),
        result,
    )
    return result.numpy()[indices]


def plan_ops_forward(
    tensor_a: Tensor, tensor_b: Tensor, tensor_c: Tensor, lib
) -> Tensor:
    def indexof(s: str):
        return ["bgoisl".index(i) for i in s]

    b, g, i, s, l = tensor_a.shape
    g, o, i, l = tensor_b.shape
    BGOISL = np.asarray([b, g, o, i, s, l], dtype=np.uint64)
    nBlock = lib.Get_MaxBlock(tensor_a)
    nThreads = lib.Get_WarpSize(tensor_a)
    ShMem = lib.Get_ShMem(tensor_a)
    ShElem = ShMem // tensor_a.element_size()
    global_fetch_size = 128
    Eff_ShElem = ShElem - nThreads * 4

    if Eff_ShElem <= 0:
        RuntimeError(f"not enough shared memory (L1) size : {ShMem} Bytes found")

    apply = np.full([len(BGOISL), 3], False, bool)
    apply[indexof("bgisl"), 0] = True
    apply[indexof("goil"), 1] = True
    apply[indexof("bgosl"), 2] = True
    BGOISL_div = np.empty(
        [case_count(0, Eff_ShElem, np.ones([3]), BGOISL, apply), len(BGOISL)],
        dtype=np.uint64,
    )  # block dimmension
    case_memo(
        0,
        Eff_ShElem,
        np.ones([3]),
        BGOISL,
        apply,
        0,
        np.empty_like(BGOISL_div[0]),
        BGOISL_div,
    )

    # block: data share over SM
    A_blk_size = BGOISL_div[..., indexof("bgisl")].prod(-1)
    B_blk_size = BGOISL_div[..., indexof("goil")].prod(-1)
    C_blk_size = BGOISL_div[..., indexof("bgosl")].prod(-1)
    valid_index = (A_blk_size + B_blk_size + C_blk_size) <= Eff_ShElem
    del A_blk_size, B_blk_size, C_blk_size
    BGOISL_div = BGOISL_div[valid_index]
    BGOISL_blk = (BGOISL + BGOISL_div - 1) // BGOISL_div  # block count

    # reduce cost with
    # loop axis over (low) bgiosl -> I -> S -> B -> O -> L -> G (high)
    # [bgiosl I] hold by warp (or block)
    # [B S L G] divide over block
    # 1. reduce write-back tensor_c (+ ignore read tensor_c)
    # 2. reduce fetch tensor_b

    fetch_size_gcd = np.gcd(tensor_a.element_size(), global_fetch_size)
    element_rescale = tensor_a.element_size() // fetch_size_gcd
    fetch_rescale = global_fetch_size // fetch_size_gcd

    BGOISL_ReScale = BGOISL.copy()
    BGOISL_div_ReScale = BGOISL_div.copy()
    BGOISL_ReScale[-1] *= element_rescale
    BGOISL_div_ReScale[..., -1] *= element_rescale

    A_axis = indexof("bgisl")
    B_axis = indexof("goil")
    C_axis = indexof("bgosl")
    A_access = memory_access_count(
        BGOISL_ReScale[A_axis],
        BGOISL_div_ReScale[..., A_axis],
        tensor_a.stride(),
        fetch_rescale,
    )
    B_access = memory_access_count(
        BGOISL_ReScale[B_axis],
        BGOISL_div_ReScale[..., B_axis],
        tensor_b.stride(),
        fetch_rescale,
    )
    C_access = memory_access_count(
        BGOISL_ReScale[C_axis],
        BGOISL_div_ReScale[..., C_axis],
        tensor_c.stride(),
        fetch_rescale,
    )

    A_access_multiple = np.where(
        (BGOISL_blk[..., indexof("ibs")] == 1).all(-1),
        1,
        BGOISL_blk[..., indexof("o")].prod(-1),
    )
    B_access_multiple = np.where(
        (BGOISL_blk[..., indexof("i")] == 1).all(-1),
        1,
        BGOISL_blk[..., indexof("bs")].prod(-1),
    )
    C_access_multiple = 1
    AccessCost = (
        A_access * A_access_multiple
        + B_access * B_access_multiple
        + C_access * C_access_multiple
    )
    Schedule = (
        np.ceil(BGOISL_blk[..., indexof("bgosl")].prod(-1) / nBlock)  # Block level
        * BGOISL_blk[..., indexof("i")[0]]  # non-shareable dimmension
        * (
            np.ceil(BGOISL_div.prod(-1) / nThreads)  # Warp level Prod-Sum ops
            + 1200  # Global memory load latency
        )
    )
    cost = AccessCost + Schedule
    index = cost.argmin()
    return torch.from_numpy(BGOISL_div[index].astype(np.int64))


def parse_expr(expression: str):
    op1, op2 = expression.split("->")
    return map(lambda x: x.strip(), [*op1.split(","), op2])


def execute_einsum(expression: str, Mat_A: Tensor, Mat_B: Tensor, Mat_C: Tensor, lib):
    def effective_dim(Expr, DIM_order):
        sum = 0
        for i, D in enumerate(DIM_order):
            if D in Expr:
                sum += 1 << i
        return sum

    # current version:
    # do not allow broadcast

    # check validation
    Mats = [Mat_A, Mat_B, Mat_C]
    Exprs = [Expr_A, Expr_B, Expr_C] = map(set, parse_expr(expression))
    assert Expr_C.issubset(Expr_A | Expr_B)
    assert all(map(lambda x, y: x.ndim == len(y), Mats, Exprs))
    DIMs = {}
    for Mat, Expr in zip(Mats, Exprs):
        for S, E in zip(Mat.shape, Expr):
            if E not in DIMs:  # or DIMs[E] == 1:
                DIMs[E] = S
            # assert S == 1 or DIMs[E] == S
            assert DIMs[E] == S

    # TODO: P0 Generalize the planning process
    DIV = dict(zip("bgoisl", plan_ops_forward(Mat_A, Mat_B, Mat_C, lib)))

    # TODO: P2 priority logic improvement
    axis_R = sorted(Expr_A & Expr_B - Expr_C)  # reduction axis
    axis_BC_A = sorted(Expr_A & Expr_C - Expr_B)  # broadcast from A
    axis_BC_B = sorted(Expr_B & Expr_C - Expr_A)  # broadcast from B
    axis_S = sorted(Expr_A & Expr_B & Expr_C)  # shared axis
    DIM_order = axis_R + axis_BC_A + axis_BC_B + axis_S

    priority = {I: i for i, I in enumerate(DIM_order)}
    Parse_A = "".join(sorted(Expr_A.copy(), key=priority.__getitem__)[::-1])
    Parse_B = "".join(sorted(Expr_B.copy(), key=priority.__getitem__)[::-1])
    Parse_C = "".join(sorted(Expr_C.copy(), key=priority.__getitem__)[::-1])
    Parses = [Parse_A, Parse_B, Parse_C]

    opt = dict(dtype=torch.int64)
    lib.PlaneDot(
        torch.einsum(f"{Expr_A}->{Parse_A}", Mat_A),
        torch.einsum(f"{Expr_B}->{Parse_B}", Mat_B),
        torch.einsum(f"{Expr_C}->{Parse_C}", Mat_C),
        torch.tensor([DIMs[D] for D in DIM_order], **opt),
        torch.tensor([DIV[D] for D in DIM_order], **opt),
        torch.tensor(len(axis_R), **opt),
        torch.tensor([np.prod(map(DIV.__getitem__, x)) for x in Parses], **opt),
        torch.tensor([effective_dim(x, DIM_order) for x in Parses], **opt),
    )
    return Mat_C


class FastPlaneDot(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a: Tensor, b: Tensor):
        assert a.is_contiguous(memory_format=torch.contiguous_format)
        assert b.is_contiguous(memory_format=torch.contiguous_format)
        assert a.dtype == b.dtype
        assert a.device == b.device
        assert a.ndim == 5 and b.ndim == 4
        Ab, Ag, Ai, As, Al = a.shape
        Bg, Bo, Bi, Bl = b.shape
        assert Ag == Bg
        assert Ai == Bi
        assert Al == Bl
        ctx.save_for_backward(a, b)
        c = a.new_empty([Ab, Ag, Bo, As, Al])
        {
            "CPU": cpu.PlaneDot_forward,
            "CUDA": lambda a, b, c: execute_einsum("bgisl,goil->bgosl", a, b, c, cuda),
            "HIP": RaiseNotImplementedError("Device type: HIP"),
        }[get_device_type(a)](a, b, c)
        c.requires_grad = a.requires_grad or b.requires_grad
        return c

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        a, b = ctx.saved_tensors
        grad_A = a.new_zeros(a.shape)
        grad_B = b.new_zeros(b.shape)
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
        return grad_A.conj(), grad_B.conj()
