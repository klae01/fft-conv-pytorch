import gc
import time
from contextlib import contextmanager
from timeit import Timer
from typing import Callable, NamedTuple, Tuple

import numpy as np
import torch
from torch import Tensor


class Benchmark(NamedTuple):
    mean: float
    std: float

    def __repr__(self):
        return f"BenchmarkResult(mean: {self.mean:.3e}, std: {self.std:.3e})"

    def __str__(self):
        return f"({self.mean:.3e} \u00B1 {self.std:.3e}) s"


@contextmanager
def measure():
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    result = dict()
    begin = time.time()
    try:
        yield result
    finally:
        torch.cuda.synchronize()
    result["time"] = time.time() - begin
    result["memory"] = torch.cuda.memory_stats()["allocated_bytes.all.peak"] / 2**30


def benchmark(
    fn: Callable, *args, num_iterations: int = 10, **kwargs
) -> Tuple[Benchmark, Benchmark]:
    time, memory = [], []
    for i in range(num_iterations):
        with measure() as r:
            fn(*args, **kwargs)
        time.append(r.get("time"))
        memory.append(r.get("memory"))
    return Benchmark(np.mean(time[1:]).item(), np.std(time[1:]).item()), Benchmark(
        np.mean(memory[1:]).item(), np.std(memory[1:]).item()
    )


def _assert_almost_equal(x: Tensor, y: Tensor) -> bool:
    abs_error = torch.abs(x - y)
    assert abs_error.mean().item() < 5e-5
    assert abs_error.max().item() < 1e-4
    return True


def _gcd(x: int, y: int) -> int:
    while y:
        x, y = y, x % y
    return x
