import numpy as np
import numba


@numba.njit()
def case_count(depth, remain, ARR):
    if depth == len(ARR):
        return 1
    sum = 0
    for i in range(1, min(ARR[depth], remain) + 1):
        sum += case_count(depth + 1, remain // i, ARR)
    return sum


@numba.njit()
def case_memo(depth, remain, ARR, index, stack_result, result):
    if depth == len(ARR):
        result[index] = stack_result
        return index + 1
    for i in range(1, min(ARR[depth], remain) + 1):
        stack_result[depth] = i
        index = case_memo(depth + 1, remain // i, ARR, index, result)
    return index


def _naive_global_memory_access(point, DIV, stride, chunksize, result):
    acc = point
    for i, I in enumerate(DIV):
        shape = [1] * len(DIV)
        shape[i] = -1
        acc = acc + np.arange(I).reshape(shape) * stride[i]
    result += len(np.unique(acc//chunksize))


def _naive_global_memory_access_loop(
    depth, point, stack_DIV, DIM, DIV, chunksize, stride, result
):
    if depth == len(DIM):
        # prev_sum = result.sum()
        _naive_global_memory_access(point, stack_DIV, stride, chunksize, result)
    else:
        for i in range(0, DIM[depth], DIV[depth]):
            stack_DIV[depth] = min(DIV[depth], DIM[depth] - i)
            _naive_global_memory_access_loop(
                depth + 1, point, stack_DIV, DIM, DIV, chunksize, stride, result
            )
            point += stride[depth] * DIV[depth]


def naive_global_memory_access(DIM, DIV, stride, chunksize):
    # ignore out of range element access
    result = np.array([0], np.uint32)
    _naive_global_memory_access_loop(
        0,
        np.int64(0),
        np.empty_like(DIV),
        DIM,
        DIV,
        np.int64(chunksize),
        np.array(stride, np.int64),
        result,
    )
    return sum(result)

