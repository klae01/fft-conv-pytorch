from functools import lru_cache, partial
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as f
from matplotlib.ticker import MaxNLocator
from tqdm import tqdm

from fft_conv_pytorch import functional
from fft_conv_pytorch.benchmark_utils import Benchmark, benchmark

to_ntuple = functional.to_ntuple


@lru_cache(maxsize=1)
def _get_conv_inputs(
    ndim: int,
    input_size: int,
    kernel_size: Union[int, Iterable[int]],
    batch_size: int = 2,
    in_channels: int = 8,
    out_channels: int = 8,
    groups: int = 8,
):
    factory_kwargs = {"requires_grad": True, "device": "cuda", "dtype": torch.float32}
    dims = ndim * [input_size]
    signal = torch.randn(batch_size, in_channels, *dims, **factory_kwargs)

    kernel_size = to_ntuple(kernel_size, n=signal.ndim - 2)
    weight = torch.randn(
        out_channels,
        in_channels // groups,
        *kernel_size,
        **factory_kwargs,
    )
    bias = torch.randn(out_channels, **factory_kwargs)

    return signal, weight, bias


def benchmark_conv(
    ndim: int,
    input_size: int,
    kernel_size: int,
    method: Callable = None,
    num_iterations: int = 10,
    **chan_opt,
):
    torch.backends.cuda.cufft_plan_cache.clear()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    signal, weight, bias = _get_conv_inputs(
        ndim=ndim, input_size=input_size, kernel_size=kernel_size, **chan_opt
    )
    return benchmark(
        partial(method, groups=chan_opt["groups"]),
        signal,
        weight,
        bias=bias,
        num_iterations=num_iterations,
    )


def benchmark_kernel_size(
    kernel_sizes: Sequence[int],
    ndim: int,
    input_size: int,
    method: Callable = None,
    num_iterations: int = 10,
    desc: str = "",
    **chan_opt,
):
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    fn = partial(
        benchmark_conv,
        ndim=ndim,
        input_size=input_size,
        method=method,
        num_iterations=num_iterations,
        **chan_opt,
    )
    return zip(
        *(fn(kernel_size=k) for k in tqdm(kernel_sizes, desc=f"{desc:<30s}", ncols=0))
    )


def _plot_benchmarks(
    benchmarks: List[Benchmark],
    config: Dict,
    ax: plt.Axes,
    linestyle: str,
    label: Optional[str] = None,
    upper_info: bool = True,
    lower_info: bool = True,
):
    xs = config["kernel_sizes"]
    ys = np.array([b.mean for b in benchmarks])
    std = np.array([b.std for b in benchmarks])
    line, *_ = ax.plot(xs, ys, linestyle, label=label)
    ax.fill_between(
        xs, ys - std, ys + std, facecolor=linestyle[-1], alpha=0.25, label="_nolegend_"
    )
    ax.set_xticks(sorted(set(xs)))
    ax.set_yscale("log")

    ndim = config["ndim"]
    if upper_info:
        ax.set_title(f"{ndim}D")
    if lower_info:
        ax.set_xlabel(f"Kernel Size ({' x '.join(['n'] * ndim)})")
    return line


def naive_conv(input, weight, *args, **kwargs):
    return getattr(f, f"conv{weight.ndim-2}d")(input, weight, *args, **kwargs)


def naive_conv_transpose(input, weight, *args, **kwargs):
    return getattr(f, f"conv_transpose{weight.ndim-2}d")(input, weight, *args, **kwargs)


if __name__ == "__main__":
    import os

    configs = [
        {
            "ndim": 1,
            "input_size": 32768,
            "num_iterations": 16,
            "kernel_sizes": [1] + list(range(256, 4096, 512)),
            "batch_size": 2,
            "in_channels": 8,
            "out_channels": 8,
            "groups": 1,
        },
        {
            "ndim": 2,
            "input_size": 512,
            "num_iterations": 16,
            "kernel_sizes": [1] + list(np.arange(4, 49, 6)),
            "batch_size": 2,
            "in_channels": 8,
            "out_channels": 8,
            "groups": 1,
        },
        {
            "ndim": 3,
            "input_size": 64,
            "num_iterations": 16,
            "kernel_sizes": [1] + list(np.arange(2, 9, 2)),
            "batch_size": 2,
            "in_channels": 8,
            "out_channels": 8,
            "groups": 1,
        },
    ]

    save_dir = os.path.join(os.path.dirname(__file__), os.path.pardir)
    fig, ax = plt.subplots(2, 3, figsize=(5 * 3, 4 * 2), squeeze=False)
    handles = dict()

    for label, (method, linestyle) in dict(
        fft_conv_transpose=[functional.fft_conv_transpose, "s--b"],
        fft_conv=[functional.fft_conv, "o-b"],
        naive_conv=[naive_conv, "o-g"],
        naive_conv_transpose=[naive_conv_transpose, "s--g"],
    ).items():
        for i, config in enumerate(configs):
            try:
                time, memory = benchmark_kernel_size(
                    **config, method=method, desc=f"{label} {config['ndim']}D"
                )
                handles[label] = _plot_benchmarks(
                    time,
                    config=config,
                    ax=ax[0, config["ndim"] - 1],
                    linestyle=linestyle,
                    label=label,
                    lower_info=False,
                )
                handles[label] = _plot_benchmarks(
                    memory,
                    config=config,
                    ax=ax[1, config["ndim"] - 1],
                    linestyle=linestyle,
                    label=label,
                    upper_info=False,
                )
            except Exception as e:
                print(e)
    ax[0, 0].set_ylabel("Execution Time (s)")
    ax[1, 0].set_ylabel("Execution Memory (GB)")
    fig.tight_layout()
    fig.subplots_adjust(right=0.82)
    fig.legend(handles.values(), handles.keys(), loc="center right")
    plt.savefig(os.path.join(save_dir, "benchmark.png"))
