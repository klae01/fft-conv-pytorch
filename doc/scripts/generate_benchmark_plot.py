from functools import lru_cache, partial
from typing import Dict, Iterable, List, Optional, Sequence, Union, Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as f
from tqdm import tqdm
from matplotlib.ticker import MaxNLocator

from fft_conv_pytorch.functional import fft_conv, fft_conv_transpose, to_ntuple
from fft_conv_pytorch.benchmark_utils import Benchmark, benchmark


def cuda_sync(func, *args, **kwargs):
    X = func(*args, **kwargs)
    torch.cuda.synchronize()
    return X


@lru_cache(maxsize=1)
def _get_conv_inputs(
    ndim: int,
    input_size: int,
    kernel_size: Union[int, Iterable[int]],
    batch_size: int = 2,
    in_channels: int = 8,
    out_channels: int = 8,
):
    dims = ndim * [input_size]
    signal = torch.randn(batch_size, in_channels, *dims)

    kernel_size = to_ntuple(kernel_size, n=signal.ndim - 2)
    weight = torch.randn(out_channels, in_channels, *kernel_size, requires_grad=True)
    bias = torch.randn(out_channels, requires_grad=True)

    return signal.cuda(), weight.cuda(), bias.cuda()


def benchmark_conv(
    ndim: int,
    input_size: int,
    kernel_size: int,
    fft: bool = True,
    method: Callable = None,
    num_iterations: int = 10,
):
    conv_fn = method if fft else getattr(f, f"{method.__name__[4:]}{ndim}d")
    signal, weight, bias = _get_conv_inputs(
        ndim=ndim, input_size=input_size, kernel_size=kernel_size
    )
    return benchmark(
        partial(cuda_sync, conv_fn),
        signal,
        weight,
        bias=bias,
        num_iterations=num_iterations,
    )


def benchmark_kernel_size(
    kernel_sizes: Sequence[int],
    ndim: int,
    input_size: int,
    fft: bool = True,
    method: Callable = None,
    num_iterations: int = 10,
    desc: str = "",
) -> List[Benchmark]:
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    fn = partial(
        benchmark_conv,
        ndim=ndim,
        input_size=input_size,
        fft=fft,
        method=method,
        num_iterations=num_iterations,
    )
    return [fn(kernel_size=k) for k in tqdm(kernel_sizes, desc=desc)]


def _plot_benchmarks(
    benchmarks: List[Benchmark],
    config: Dict,
    ax: plt.Axes,
    color: str,
    linestyle: str,
    label: Optional[str] = None,
):
    xs = config["kernel_sizes"]
    ys = np.array([b.mean * 1000 for b in benchmarks])
    std = np.array([b.std * 1000 for b in benchmarks])
    ax.plot(xs, ys, color, linestyle=linestyle, label=label)
    ax.fill_between(
        xs, ys - std, ys + std, facecolor=color, alpha=0.25, label="_nolegend_"
    )

    ndim = config["ndim"]
    ax.set_title(f"{ndim}D")
    kernel_size_str = "(" + " x ".join(["n"] * ndim) + ")"
    ax.set_xlabel(f"Kernel Size {kernel_size_str}")
    ax.set_yscale("log")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


if __name__ == "__main__":
    import os

    configs = [
        {
            "ndim": 1,
            "input_size": 4096,
            "num_iterations": 256,
            "kernel_sizes": np.arange(64, 513, 64),
        },
        {
            "ndim": 2,
            "input_size": 512,
            "num_iterations": 16,
            "kernel_sizes": np.arange(4, 49, 6),
        },
        {
            "ndim": 3,
            "input_size": 64,
            "num_iterations": 16,
            "kernel_sizes": np.arange(2, 15, 2),
        },
    ]

    save_dir = os.path.join(os.path.dirname(__file__), os.path.pardir)
    fig, ax = plt.subplots(1, 3, figsize=(4 * len(configs), 4), squeeze=False)

    for method in [fft_conv_transpose, fft_conv]:
        for i, config in enumerate(configs):
            work_type = method.__name__[4:]
            linestyle = [None, "--"]["transpose" in work_type]
            fft = benchmark_kernel_size(
                fft=True, **config, method=method, desc=f"FFT {config['ndim']}D"
            )
            _plot_benchmarks(
                fft,
                config=config,
                ax=ax[0, config["ndim"] - 1],
                color="r",
                linestyle=linestyle,
                label=f"fft_{work_type}",
            )

            direct = benchmark_kernel_size(
                fft=False, **config, method=method, desc=f"Direct {config['ndim']}D"
            )
            _plot_benchmarks(
                direct,
                config=config,
                ax=ax[0, config["ndim"] - 1],
                color="b",
                linestyle=linestyle,
                label=f"native_{work_type}",
            )
    ax[0, 0].set_ylabel("Execution Time (ms)")
    ax[0, 2].legend()
    plt.savefig(os.path.join(save_dir, "benchmark.png"))
