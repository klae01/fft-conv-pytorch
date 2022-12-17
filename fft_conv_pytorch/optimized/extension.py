import itertools
from functools import lru_cache
from typing import Dict, Tuple, List

import numba
import numpy as np
import torch
from torch import Tensor

from .lib import cpu, cuda


FETCH_SIZE_LIMIT = 2**16
FETCH_INDEXING_DTYPE = torch.int16


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
    for x in range(1, min(limit, SQ) + 1):
        yield x
    for x in range(SQ - 1 if SQ == (N + SQ - 1) // SQ else SQ, (N - 1) // limit, -1):
        yield (N + x - 1) // x


@numba.njit()
def case_count(depth, max_ops, usage, ARR, apply):
    if depth == len(ARR):
        return 1
    sum = 0
    remain = min(
        (max_ops - usage[~apply[depth]].sum()) // usage[apply[depth]].sum(),
        FETCH_SIZE_LIMIT // usage[apply[depth]].max(),
    )
    if remain > 0:
        tmp_usage = usage.copy()
        for i in case_generate(ARR[depth], remain):
            tmp_usage[apply[depth]] = i * usage[apply[depth]]
            sum += case_count(depth + 1, max_ops, tmp_usage, ARR, apply)
    return sum


@numba.njit()
def case_memo(depth, max_ops, usage, ARR, apply, index, stack_result, result):
    if depth == len(ARR):
        result[index] = stack_result
        return index + 1
    remain = min(
        (max_ops - usage[~apply[depth]].sum()) // usage[apply[depth]].sum(),
        FETCH_SIZE_LIMIT // usage[apply[depth]].max(),
    )
    if remain > 0:
        tmp_usage = usage.copy()
        for i in case_generate(ARR[depth], remain):
            stack_result[depth] = i
            tmp_usage[apply[depth]] = i * usage[apply[depth]]
            index = case_memo(
                depth + 1, max_ops, tmp_usage, ARR, apply, index, stack_result, result
            )
    return index


@torch.no_grad()
def memory_access_count(
    DIM: np.ndarray, DIV: np.ndarray, stride: Tuple[int], chunksize: int, offset: int
):
    # bottom up, memory shift 0 ~ chunksize, all case cache
    # ignore out of range
    # TODO: P1, consider offset
    assert DIM.ndim == 1
    u_DIV, indices = np.unique(DIV, axis=0, return_inverse=True)

    result = torch.zeros([u_DIV.shape[0]], dtype=torch.float32)
    cpu.AccessCost(
        torch.tensor(len(DIM)).type(torch.int64),
        torch.tensor(chunksize).type(torch.int64),
        torch.tensor(offset).type(torch.int64),
        torch.from_numpy(DIM.astype(np.int64)),
        torch.tensor(stride).type(torch.int64),
        torch.from_numpy(u_DIV.astype(np.int64)),
        result,
    )
    return result.numpy()[indices]


def parse_expr(expression: str):
    op1, op2 = expression.split("->")
    return map(lambda x: x.strip(), [*op1.split(","), op2])


def axis_classify(Expr_A: str, Expr_B: str, Expr_C: str):
    Expr_A, Expr_B, Expr_C = map(set, [Expr_A, Expr_B, Expr_C])
    return map(
        lambda x: "".join(sorted(x)),
        [
            Expr_A & Expr_B - Expr_C,
            Expr_A & Expr_C - Expr_B,
            Expr_B & Expr_C - Expr_A,
            Expr_A & Expr_B & Expr_C,
        ],
    )


def axis_size_matching_validation(*MAPs: List[Dict]):
    def _sub(MAP_X: Dict, MAP_Y: Dict):
        for k in MAP_X.keys():
            if k in MAP_Y:
                assert MAP_X[k] == MAP_Y[k]

    [_sub(*I) for I in itertools.combinations(MAPs, 2)]


def axis_validation(
    Expr_A: str,
    Expr_B: str,
    Expr_C: str,
    a_shape: Tuple[int],
    b_shape: Tuple[int],
    c_shape: Tuple[int],
):
    # allow up to 32 dimension
    assert max(map(len, [Expr_A, Expr_B, Expr_C])) <= 32
    # do not allow trace
    assert all(map(lambda x: len(x) == len(set(x)), [Expr_A, Expr_B, Expr_C]))
    # do not allow broadcast or unmatched length
    axis_size_matching_validation(
        dict(zip(Expr_A, a_shape)),
        dict(zip(Expr_B, b_shape)),
        dict(zip(Expr_C, c_shape)),
    )


@lru_cache(2**20)
def __plan_ops(
    expression: str,
    nBlock: int,
    nThreads: int,
    ShMem: int,
    element_size: int,
    a_shape: Tuple[int],
    b_shape: Tuple[int],
    c_shape: Tuple[int],
    a_stride: Tuple[int],
    b_stride: Tuple[int],
    c_stride: Tuple[int],
    a_storage_offset: int,
    b_storage_offset: int,
    c_storage_offset: int,
) -> np.ndarray:
    def indexof(s: str):
        nonlocal AXIS_ORDER
        return [AXIS_ORDER.index(i) for i in s]

    Expr_A, Expr_B, Expr_C = parse_expr(expression)
    axis_validation(Expr_A, Expr_B, Expr_C, a_shape, b_shape, c_shape)
    axis_size = dict(
        [
            *zip(Expr_A, a_shape),
            *zip(Expr_B, b_shape),
            *zip(Expr_C, c_shape),
        ]
    )
    AXIS_ORDER = "".join(axis_size.keys())
    SIZE = np.asarray([axis_size[i] for i in AXIS_ORDER], dtype=np.int64)

    # reduction axis ( A & B axis )
    # broadcast from A ( A & C axis )
    # broadcast from B ( B & C axis )
    # shared axis ( A & B & C axis )
    axis_R, axis_A, axis_B, axis_S = axis_classify(Expr_A, Expr_B, Expr_C)
    ShElem = ShMem // element_size
    global_fetch_size = 128
    Eff_ShElem = ShElem - nThreads * 4

    if Eff_ShElem <= 0:
        RuntimeError(f"not enough shared memory (L1) size : {ShMem} Bytes found")

    apply = np.full([len(SIZE), 3], False, bool)
    apply[indexof(Expr_A), 0] = True
    apply[indexof(Expr_B), 1] = True
    apply[indexof(Expr_C), 2] = True
    # block: data share over SM
    # block dimmension
    DIV = np.empty(
        [
            case_count(0, Eff_ShElem, np.ones([3], dtype=np.int64), SIZE, apply),
            len(SIZE),
        ],
        dtype=np.int64,
    )
    case_memo(
        0,
        Eff_ShElem,
        np.ones([3], dtype=np.int64),
        SIZE,
        apply,
        0,
        np.empty_like(DIV[0], dtype=np.int64),
        DIV,
    )
    BLK = (SIZE + DIV - 1) // DIV  # block count

    # reduce cost with
    # loop axis over (low) bgiosl -> I -> S -> B -> O -> L -> G (high)
    # [bgiosl I] hold by warp (or block)
    # [B S L G] divide over block
    # 1. reduce write-back tensor_c (+ ignore read tensor_c)
    # 2. reduce fetch tensor_b

    fetch_size_gcd = np.gcd(element_size, global_fetch_size)
    element_rescale = element_size // fetch_size_gcd
    fetch_rescale = global_fetch_size // fetch_size_gcd

    ReScale = lambda C: lambda X: X if element_rescale == 0 else C(X)
    RS_SIZE = ReScale(lambda X: np.concatenate([X, [element_rescale]], dtype=X.dtype))
    RS_DIV = ReScale(
        lambda X: np.concatenate(
            [X, np.broadcast_to([element_rescale], [X.shape[0], 1])],
            axis=-1,
            dtype=X.dtype,
        )
    )
    RS_Stride = ReScale(lambda X: tuple([x * element_rescale for x in X] + [1]))
    RS_Chunk = ReScale(lambda X: X)
    RS_Offset = ReScale(lambda X: X * element_rescale)

    A_access = memory_access_count(
        RS_SIZE(SIZE[indexof(Expr_A)]),
        RS_DIV(DIV[..., indexof(Expr_A)]),
        RS_Stride(a_stride),
        RS_Chunk(fetch_rescale),
        RS_Offset(a_storage_offset),
    )
    B_access = memory_access_count(
        RS_SIZE(SIZE[indexof(Expr_B)]),
        RS_DIV(DIV[..., indexof(Expr_B)]),
        RS_Stride(b_stride),
        RS_Chunk(fetch_rescale),
        RS_Offset(b_storage_offset),
    )
    C_access = memory_access_count(
        RS_SIZE(SIZE[indexof(Expr_C)]),
        RS_DIV(DIV[..., indexof(Expr_C)]),
        RS_Stride(c_stride),
        RS_Chunk(fetch_rescale),
        RS_Offset(c_storage_offset),
    )

    A_access_multiple = np.where(
        (BLK[..., indexof(axis_R + axis_A)] == 1).all(-1),
        1,
        BLK[..., indexof(axis_B)].prod(-1),
    )
    B_access_multiple = np.where(
        (BLK[..., indexof(axis_R)] == 1).all(-1),
        1,
        BLK[..., indexof(axis_A)].prod(-1),
    )
    C_access_multiple = 1
    AccessCost = (
        A_access * A_access_multiple
        + B_access * B_access_multiple
        + C_access * C_access_multiple
    )
    Schedule = (
        np.ceil(BLK[..., indexof(Expr_C)].prod(-1) / nBlock)  # Block level
        * BLK[..., indexof(axis_R)].prod(-1)  # non-shareable dimmension
        * (
            np.ceil(DIV.prod(-1) / nThreads)  # Warp level Prod-Sum ops
            + 1200  # Global memory load latency
        )
    )
    cost = AccessCost + Schedule
    index = cost.argmin()
    return cost[index], SIZE, DIV[index], BLK[index], AXIS_ORDER


def plan_ops(
    expression: str, tensor_a: Tensor, tensor_b: Tensor, tensor_c: Tensor, lib
) -> Tensor:
    def kerf_info(expr):
        nonlocal DIM, DIV, AXIS
        flag = 0
        case = []
        for i, j in enumerate(map(AXIS.index, expr)):
            remain = DIM[j] % DIV[j]
            if remain != 0:
                flag += 1 << i
                case.append([DIV[i], remain])
            else:
                case.append(DIV[i])
        return flag, case

    Expr_A, Expr_B, Expr_C = parse_expr(expression)

    COST, DIM, DIV, BLK, AXIS = __plan_ops(
        expression,
        lib.Get_MaxBlock(tensor_a),
        lib.Get_WarpSize(tensor_a),
        lib.Get_ShMem(tensor_a),
        tensor_a.element_size(),
        tuple(tensor_a.shape),
        tuple(tensor_b.shape),
        tuple(tensor_c.shape),
        tuple(tensor_a.stride()),
        tuple(tensor_b.stride()),
        tuple(tensor_c.stride()),
        tensor_a.storage_offset(),
        tensor_b.storage_offset(),
        tensor_c.storage_offset(),
    )
    return torch.tensor([DIV[AXIS.index(i)] for i in "bgoisl"])


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
    Sub_A, Sub_B, Sub_C = parse_expr(expression)
    Subs = [Sub_A, Sub_B, Sub_C]
    Expr_A, Expr_B, Expr_C = map(set, Subs)
    Exprs = [Expr_A, Expr_B, Expr_C]
    assert Expr_C.issubset(Expr_A | Expr_B)
    assert all(map(lambda x, y: x.ndim == len(y), Mats, Exprs))
    DIMs = {}
    for Mat, Expr in zip(Mats, Subs):
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
        torch.einsum(f"{Sub_A}->{Parse_A}", Mat_A),
        torch.einsum(f"{Sub_B}->{Parse_B}", Mat_B),
        torch.einsum(f"{Sub_C}->{Parse_C}", Mat_C),
        torch.tensor([DIMs[D] for D in DIM_order[::-1]], **opt),
        torch.tensor([DIV[D] for D in DIM_order[::-1]], **opt),
        torch.tensor(len(axis_R), **opt),
        torch.tensor([np.prod([DIV[i] for i in x]) for x in Parses], **opt),
        torch.tensor([effective_dim(x, DIM_order) for x in Parses], **opt),
    )


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
