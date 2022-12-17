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


@numba.njit()
def _naive_global_memory_access(depth, point, chunk_id, DIV, stride, chunksize, result):
    if depth == len(DIV):
        if chunk_id != point // chunksize:
            chunk_id = point // chunksize
            result[chunk_id] += 1
        return chunk_id
    for i in range(0, DIV[depth]):
        chunk_id = _naive_global_memory_access(
            depth + 1, point, chunk_id, DIV, stride, chunksize, result
        )
        point += stride[depth]
    return chunk_id


@numba.njit()
def _naive_global_memory_access_loop(
    depth, point, stack_DIV, DIM, DIV, chunksize, stride, result
):
    if depth == len(DIM):
        prev_sum = result.sum()
        _naive_global_memory_access(0, point, -1, stack_DIV, stride, chunksize, result)
    else:
        for i in range(0, DIM[depth], DIV[depth]):
            stack_DIV[depth] = min(DIV[depth], DIM[depth] - i)
            _naive_global_memory_access_loop(
                depth + 1, point, stack_DIV, DIM, DIV, chunksize, stride, result
            )
            point += stride[depth] * DIV[depth]


def naive_global_memory_access(DIM, DIV, chunksize):
    # ignore out of range element access
    result = np.zeros(int((DIM.prod() + chunksize - 1) // chunksize), np.uint32)
    stride = DIM[::-1].cumprod()[::-1]
    stride = np.roll(stride, -1)
    stride[-1] = 1
    _naive_global_memory_access_loop(
        0,
        np.int64(0),
        np.empty_like(DIV),
        DIM,
        DIV,
        np.int64(chunksize),
        stride,
        result,
    )
    return sum(result)


@numba.njit()
def loopbincount(data, minlength, loop, remain):
    assert remain >= 0
    if loop and remain:
        return np.bincount(data, minlength=minlength) * loop + np.bincount(
            data[:remain], minlength=minlength
        )
    elif loop:
        return np.bincount(data, minlength=minlength) * loop
    return np.bincount(data[:remain], minlength=minlength)


def _fast_global_memory_access(depth, DIM, DIV, stride, chunksize, array):
    cache_access, num_access = array[(depth) % 2], array[(depth + 1) % 2]
    cache_remain, num_remain = array[(depth) % 2 + 2], array[(depth + 1) % 2 + 2]
    if depth == len(DIM):
        num_access[...] = 1
        num_remain[::-1] = np.arange(chunksize)
        return
    # cache_access[i] : shift = i, memory access count at [depth:ndim]
    # cache_remain[i] = 0 : start from new
    _fast_global_memory_access(depth + 1, DIM, DIV, stride, chunksize, array)
    # if depth >= 3:
    #     print(depth, cache_access, cache_remain)
    # DIM[depth] // chunk size ; DIM[depth] % chunk size ;
    index = np.arange(chunksize) * stride[depth] % chunksize
    loop = np.lcm(DIV[depth], chunksize)

    # for shift in range(chunksize):
    #     for i in range(0, DIM[depth], DIV[depth]):
    #         for j in range(i, i+min(DIV[depth], DIM[depth] - i)):
    #             index = (shift + j * stride[depth]) % chunksize
    #             num_access[shift] += cache_access[index]
    num_access[:chunksize] = np.convolve(
        np.tile(cache_access, 2),
        loopbincount(
            index,
            minlength=chunksize,
            loop=DIM[depth] // chunksize,
            remain=DIM[depth] % chunksize,
        )[::-1],
        "valid",
    )[:chunksize]

    # for shift in range(chunksize):
    #     for i in range(0, DIM[depth]-1):
    #         if (i + 1) % DIV[depth]:
    #             index = (shift + i * stride[depth]) % chunksize
    #             if cache_remain[index] >= stride[depth]:
    #                 num_access[shift] -= 1
    num_access[:chunksize] -= np.convolve(
        np.tile(cache_remain >= stride[depth], 2),
        (
            loopbincount(
                index,
                minlength=chunksize,
                loop=(DIM[depth] - 1) // chunksize,
                remain=(DIM[depth] - 1) % chunksize,
            )
            - loopbincount(
                np.arange(DIV[depth] - 1, loop, DIV[depth]) * stride[depth] % chunksize,
                minlength=chunksize,
                loop=((DIM[depth] - 1) // loop),
                remain=((DIM[depth] - 1) % loop + 1) // DIV[depth],
            )
        )[::-1],
        "valid",
    )[:chunksize]

    num_remain[:chunksize] = np.maximum(
        0, cache_remain[index] - stride[depth] * (DIM[depth] - DIV[depth])
    )


def fast_global_memory_access(DIM, DIV, chunksize):
    # bottom up, memory shift 0 ~ chunksize, all case cache
    # ignore out of range
    stride = DIM[::-1].cumprod()[::-1]
    stride = np.roll(stride, -1)
    stride[-1] = 1
    DIM = np.array(DIM, np.int64)
    DIV = np.array(DIV, np.int64)
    stride = np.array(stride, np.int64)
    chunksize = np.int64(chunksize)
    array = np.zeros([4, chunksize], np.int64)
    _fast_global_memory_access(0, DIM, DIV, stride, chunksize, array)
    return array[1, 0]
