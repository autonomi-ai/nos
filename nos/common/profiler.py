import contextlib
import datetime
import gc
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Union

import pandas as pd
import psutil
import torch
from torch.profiler import ProfilerActivity, profile  # noqa
from torch.profiler import record_function as _record_function  # noqa

from nos.common import tqdm
from nos.common.system import get_system_info, has_gpu
from nos.constants import NOS_CACHE_DIR
from nos.logging import logger
from nos.version import __version__


NOS_PROFILE_DIR = NOS_CACHE_DIR / "profile"
NOS_PROFILE_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_PROFILER_SCHEDULE = torch.profiler.schedule(wait=10, warmup=10, active=80, repeat=0)


@dataclass
class ExecutionStats:
    """Execution statistics."""

    num_iterations: int
    """Number of iterations."""
    total_ms: float
    """Total time in milliseconds."""
    cpu_utilization: float
    """CPU utilization."""
    gpu_utilization: Union[None, float]
    """GPU utilization."""

    @property
    def fps(self):
        return self.num_iterations / (self.total_ms * 1e3)


class profile_execution:
    """Context manager for profiling execution."""

    def __init__(self, name: str, iterations: int = None, duration: float = None):
        if iterations is None and duration is None:
            raise ValueError("Either `iterations` or `duration` must be specified.")
        self.iterations = iterations
        self.duration = duration
        self.name = f"{name}::execution"
        self.iterator = None
        self.execution_stats = None

    def __repr__(self) -> str:
        return (
            f"""{self.__class__.__name__} """
            f"""(name={self.name}, iterations={self.iterations}, duration={self.duration}, """
            f"""stats={self.execution_stats})"""
        )

    def __enter__(self) -> "profile_execution":
        """Start profiling execution."""
        # Note (spillai): The first call to `cpu_percent` with `interval=None` starts
        # capturing the CPU utilization. The second call in `__exit__` returns
        # the average CPU utilization over the duration of the context manager.
        _ = psutil.cpu_percent(interval=None)
        self.iterator = (
            tqdm(duration=self.duration, desc=self.name)
            if self.duration
            else tqdm(total=self.iterations, desc=self.name)
        )
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        """Stop profiling."""
        end_t = time.time()
        cpu_util = psutil.cpu_percent(interval=None)
        try:
            # TOFIX (spillai): This will be fixed with torch 2.1
            gpu_util = torch.cuda.utilization(int(os.getenv("CUDA_VISIBLE_DEVICES", None)))
        except Exception:
            gpu_util = None
        self.execution_stats = ExecutionStats(
            self.iterator.n, (end_t - self.iterator.start_t) * 1e3, cpu_util, gpu_util
        )

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.iterator)


class profile_memory:
    """Context manager for profiling memory usage (GPU/CPU)."""

    def __init__(self, name: str):
        self.name = name
        self.memory_stats = {}

    def __repr__(self) -> str:
        return f"""{self.__class__.__name__} (name={self.name}, stats={self.memory_stats})"""

    def __enter__(self) -> "profile_memory":
        """Start profiling GPU memory usage."""
        try:
            free, total = torch.cuda.mem_get_info()
            self.memory_stats[f"{self.name}::memory_gpu::pre"] = total - free
        except Exception:
            self.memory_stats[f"{self.name}::memory_gpu::pre"] = None
        finally:
            free, total = psutil.virtual_memory().available, psutil.virtual_memory().total
            self.memory_stats[f"{self.name}::memory_cpu::pre"] = total - free
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        """Stop profiling GPU memory usage."""
        try:
            free, total = torch.cuda.mem_get_info()
            self.memory_stats[f"{self.name}::memory_gpu::post"] = total - free
        except Exception:
            self.memory_stats[f"{self.name}::memory_gpu::post"] = None
        finally:
            free, total = psutil.virtual_memory().available, psutil.virtual_memory().total
            self.memory_stats[f"{self.name}::memory_cpu::post"] = total - free
        return

    def memory_usage(self):
        return self.memory_stats


class profiler_record:
    """Profile record for capturing profiling data (execution profile, memory profile, etc.)."""

    def __init__(self, namespace: str, **kwargs):
        self.namespace = namespace
        self.kwargs = kwargs
        self.profile_data = {}

    @contextlib.contextmanager
    def profile_execution(self, name: str = None, iterations: int = None, duration: float = None) -> profile_execution:
        """Context manager for profiling execution time."""
        with profile_execution(f"{name}", iterations=iterations, duration=duration) as prof:
            yield prof
        self.profile_data[prof.name] = prof.execution_stats.__dict__
        print(prof)

    @contextlib.contextmanager
    def profile_memory(self, name: str = None) -> profile_memory:
        """Context manager for profiling memory usage."""
        with profile_memory(f"{name}") as prof:
            yield prof
        # TODO (spillai): This is to avoid nested namespaces in the profile data dict.
        self.profile_data.update(prof.memory_usage())
        print(prof)

    def as_dict(self) -> Dict[str, Any]:
        """Return a dictionary representation of the profiler record."""
        return {
            "namespace": self.namespace,
            "profile_data": self.profile_data,
            **self.kwargs,
        }


@dataclass
class Profiler:
    """NOS profiler as a context manager."""

    records: List[profiler_record] = field(default_factory=list)
    """List of profiler records."""

    def __enter__(self) -> "Profiler":
        """Start profiling benchmarks, clearing all torch.cuda cache/stats ."""
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.reset_accumulated_memory_stats()
        except Exception:
            pass
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        """Stop profiling benchmarks."""
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.reset_accumulated_memory_stats()
        except Exception:
            pass
        gc.collect()
        return

    def add(self, name: str, **kwargs) -> profiler_record:
        """Add a profiler record."""
        if len(self.records) > 0 and self.records[0].kwargs.keys() != kwargs.keys():
            raise ValueError("Adding a new record with different kwargs is not supported.")
        self.records.append(profiler_record(name, **kwargs))
        return self.records[-1]

    def as_dict(self) -> Dict[str, Any]:
        """Return a dictionary representation of the profiler."""
        return {
            "date": datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            "nos_version": __version__,
            "sysinfo": get_system_info(docker=True, gpu=True),
            "records": [r.as_dict() for r in self.records],
        }

    def __repr__(self) -> str:
        """Return a string representation of the profiler."""
        return json.dumps(self.as_dict(), indent=4)

    def save(self, filename: Union[Path, str]) -> None:
        """Save profiled results to a file."""
        with open(str(filename), "w") as f:
            json.dump(self.as_dict(), f, indent=4)

    @classmethod
    def load(cls, filename: Union[Path, str]) -> Dict[str, Any]:
        """Load profiled results."""
        with open(str(filename), "r") as f:
            return json.load(f)

    @classmethod
    def load_metadata(cls, filename: Union[Path, str]) -> pd.DataFrame:
        """Load profiled metadata."""
        data = cls.load(filename)
        _ = data.pop("records")
        return data

    @classmethod
    def load_records(cls, filename: Union[Path, str]) -> pd.DataFrame:
        """Load profiled records as a dataframe."""
        data = cls.load(filename)
        records = data.pop("records")
        return pd.json_normalize(records)


@contextlib.contextmanager
def _profiler(
    schedule: torch.profiler.schedule = None, profile_memory: bool = False, export_chrome_trace: bool = False
):
    """Default NOS profiler as a context manager."""
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA] if has_gpu() else [ProfilerActivity.CPU]
    with profile(activities=activities, schedule=schedule, profile_memory=profile_memory) as prof:
        prof._sys_info = get_system_info()
        prof._mem_prof = {}
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
