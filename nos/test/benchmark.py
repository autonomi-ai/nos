from typing import Callable

from torch.utils.benchmark import Timer


def run_benchmark(func: Callable, num_iters: int = 100):

    # Time the model
    timer = Timer(stmt="func()", globals={"func": func})
    result = timer.timeit(number=num_iters)

    # Get the average time per iteration in milliseconds
    avg_time = result.mean * 1000
    return round(avg_time, 2)
