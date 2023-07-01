import contextlib
import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from torch.profiler import ProfilerActivity, profile  # noqa
from torch.profiler import record_function as _record_function  # noqa

from nos.common.system import has_gpu
from nos.constants import NOS_CACHE_DIR
from nos.logging import logger


NOS_PROFILE_DIR = NOS_CACHE_DIR / "profile"
NOS_PROFILE_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_PROFILER_SCHEDULE = torch.profiler.schedule(wait=10, warmup=10, active=80, repeat=0)


@contextlib.contextmanager
def record_function(name: str, args: Optional[str] = None):
    """Default NOS record_function."""

    with _record_function(name, args) as rf:
        rf._gpu_memory_usage = {}
        rf._gpu_memory_usage[f"{name}::_before"] = torch.cuda.mem_get_info()
        yield rf
        rf._gpu_memory_usage[f"{name}::_after"] = torch.cuda.mem_get_info()


@contextlib.contextmanager
def profiler(
    schedule: torch.profiler.schedule = None, profile_memory: bool = False, export_chrome_trace: bool = False
):
    """Default NOS profiler as a context manager."""
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA] if has_gpu() else [ProfilerActivity.CPU]
    with profile(activities=activities, schedule=schedule, profile_memory=profile_memory) as prof:
        yield prof

    if export_chrome_trace:
        profile_path = Path(NOS_PROFILE_DIR) / f"chrome_trace_{datetime.datetime.utcnow().isoformat()}.json"
        prof.export_chrome_trace(str(profile_path))
        logger.debug(f"Profile saved to {profile_path}")


def profiler_table(prof, namespace: str = None, verbose: bool = False) -> pd.DataFrame:
    """Convert profiler output to a DataFrame."""
    events_df = pd.DataFrame([e.__dict__ for e in prof.key_averages()])
    if verbose:
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    if namespace:
        events_df = events_df[events_df.key.str.contains(f"{namespace}::")]
    columns = ["key", "count", "cpu_time_total", "self_cpu_memory_usage", "cuda_time_total", "self_cuda_memory_usage"]
    events_df = events_df[columns]
    events_df["cpu_time_avg"] = events_df["cpu_time_total"] / events_df["count"] * 1e-3
    events_df["cuda_time_avg"] = events_df["cuda_time_total"] / events_df["count"] * 1e-3
    events_df["cpu_fps"] = 1000.0 / events_df["cpu_time_avg"]
    events_df["cuda_fps"] = 1000.0 / events_df["cuda_time_avg"]
    events_df["self_cpu_memory_usage_avg"] = events_df["self_cpu_memory_usage"] / events_df["count"]
    events_df["self_cuda_memory_usage_avg"] = events_df["self_cuda_memory_usage"] / events_df["count"]
    return events_df.set_index("key")
