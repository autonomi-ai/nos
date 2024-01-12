import contextlib
import datetime
import gc
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Union

import pandas as pd
import psutil
import torch
from rich import print
from rich.tree import Tree
from torch.profiler import ProfilerActivity, profile  # noqa
from torch.profiler import record_function as _record_function  # noqa

from nos import hub
from nos.common import tqdm
from nos.common.system import get_system_info, has_gpu
from nos.constants import NOS_CACHE_DIR
from nos.logging import logger
from nos.version import __version__


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


@dataclass
class ProfilerMetadata:
    """Additional profiler metadata to be written to the profile JSON."""

    version: str = __version__
    """NOS version."""

    date: str = field(default_factory=lambda: datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"))
    """Date of profiling."""

    sysinfo: Dict[str, Any] = field(default_factory=lambda: get_system_info(docker=True, gpu=True))
    """System information."""


class profile_execution:
    """Context manager for profiling execution."""

    def __init__(self, name: str, iterations: int = None, duration: float = None, verbose: bool = True):
        if iterations is None and duration is None:
            raise ValueError("Either `iterations` or `duration` must be specified.")
        self.iterations = iterations
        self.duration = duration
        self.name = f"{name}::execution"
        self.iterator = None
        self.execution_stats = None
        self.verbose = verbose

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
            tqdm(duration=self.duration, desc=self.name, disable=not self.verbose)
            if self.duration
            else tqdm(total=self.iterations, desc=self.name, disable=not self.verbose)
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
        self.prof = {}

    @contextlib.contextmanager
    def profile_execution(self, name: str = None, iterations: int = None, duration: float = None) -> profile_execution:
        """Context manager for profiling execution time."""
        with profile_execution(f"{name}", iterations=iterations, duration=duration) as prof:
            yield prof
        self.prof[prof.name] = prof.execution_stats.__dict__

    @contextlib.contextmanager
    def profile_memory(self, name: str = None) -> profile_memory:
        """Context manager for profiling memory usage."""
        with profile_memory(f"{name}") as prof:
            yield prof
        # TODO (spillai): This is to avoid nested namespaces in the profile data dict.
        self.prof.update(prof.memory_usage())

    def update(self, key: str, value: Any) -> None:
        self.prof[key] = value

    def as_dict(self) -> Dict[str, Any]:
        """Return a dictionary representation of the profiler record."""
        return {
            "namespace": self.namespace,
            "prof": self.prof,
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

    def add(self, namespace: str, **kwargs) -> profiler_record:
        """Add a profiler record."""
        if len(self.records) > 0 and self.records[0].kwargs.keys() != kwargs.keys():
            raise ValueError("Adding a new record with different kwargs is not supported.")
        self.records.append(profiler_record(namespace, **kwargs))
        return self.records[-1]

    def as_df(self) -> pd.DataFrame:
        """Return a dataframe representation of the profiled result."""
        metadata = ProfilerMetadata()
        df = pd.json_normalize([r.as_dict() for r in self.records]).assign(
            date=metadata.date, version=metadata.version
        )
        return df

    def __repr__(self) -> str:
        """Return a string representation of the profiler."""
        return f"""{self.__class__.__name__}\n{self.as_df()}"""

    def save(self, filename: Union[Path, str]) -> None:
        """Save profiled results to JSON."""
        self.as_df().to_json(str(filename), orient="records", indent=4)

    @classmethod
    def load(cls, filename: Union[Path, str]) -> pd.DataFrame:
        """Load profiled results."""
        return pd.read_json(str(filename), orient="records")


class ModelProfileRequest:
    def __init__(self, model_id: str, method: str, get_inputs: Callable, **kwargs):
        self.model_id = model_id
        self.method = method
        self.get_inputs = get_inputs
        self.kwargs = kwargs

    def __repr__(self) -> str:
        return f"BenchmarkModel (model_id={self.model_id}, method={self.method}, kwargs={self.kwargs})"


@dataclass
class ModelProfiler:
    """Benchmark profiler.

    Usage:
        >>> from nos.cli.benchmark import ModelProfiler
        >>> profiler = ModelProfiler(models=[...])
    """

    mode: str = "full"
    """Benchmark mode (full, memory, execution)."""
    runtime: str = "gpu"
    """Runtime (cpu, gpu)."""
    requests: List[ModelProfileRequest] = field(default_factory=list)
    """Model requests to benchmark."""
    prof: Profiler = None
    """Profiler used for benchmarking."""
    device_id: int = -1
    """Device ID."""
    device_name: str = None
    """Device name."""
    device: torch.device = None
    """Torch Device to run benchmark."""

    def __repr__(self) -> str:
        from io import StringIO

        from rich.console import Console

        f = StringIO()
        console = Console(file=f, force_terminal=True)
        tree = Tree(
            f"🛠️  BenchmarkProfiler  (models={len(self.requests)}, device_name={self.device_name}, device={self.device})"
        )
        for model in self.requests:
            tree.add(f"{model}")
        console.print(tree)
        return f.getvalue()

    def __post_init__(self) -> None:
        """Setup the device and profiler."""

        # Get system info
        sysinfo = get_system_info(docker=True, gpu=True)

        # GPU / CPU acceleration
        if torch.cuda.is_available():
            assert has_gpu(), "CUDA is not available."
            assert sysinfo["gpu"] is not None, "No CUDA devices found."

            # Print GPU devices before running benchmarks
            gpu_devices = sysinfo["gpu"]["devices"]
            gpu_devices_df = pd.json_normalize(gpu_devices)
            unique_gpu_devices = gpu_devices_df.device_name.unique()
            print(f"Found GPU devices: {len(gpu_devices_df)}, unique: {len(unique_gpu_devices)}")
            if len(unique_gpu_devices) > 1:
                print(f"Multiple devices detected, selecting {unique_gpu_devices[0]}.")

            print(gpu_devices_df.to_markdown())

            # Select the appropriate device, and retrieve its name
            self.device_name = gpu_devices_df.device_name.iloc[self.device_id]
            self.device_name = self.device_name.replace(" ", "-").lower()
            self.device = torch.device(f"cuda:{self.device_id}")
            assert self.device.index == self.device_id, "Device index mismatch."
            assert self.device.type == "cuda", "Device type mismatch."
        else:
            # CPU device
            self.device_id, self.device_name = -1, "cpu"
            self.device = torch.device("cpu")

    def add(self, request: ModelProfileRequest) -> None:
        """Request a model to be added to the profiler."""
        self.requests.append(request)

    def _benchmark(self, request: ModelProfileRequest) -> None:
        """Benchmark / profile a specific model."""

        print()
        tree = Tree(
            f"🔥 [bold white]Profiling (name={request.model_id}, device={self.device_name}, kwargs={request.kwargs})[/bold white]"
        )

        # Add a new record to profile
        record = self.prof.add(
            namespace=f"nos::{request.model_id}",
            model_id=request.model_id,
            method=request.method,
            runtime=self.runtime,
            device_name=self.device_name,
            device_type=self.device.type,
            device_index=self.device.index,
        )
        with record.profile_memory("wrap"):
            try:
                # Initialize (profile memory)
                with record.profile_memory("init") as prof:
                    model = hub.load(request.model_id)
                    predict = getattr(model, request.method)
                tree.add(f"[bold green]✓[/bold green] {prof}).")

                batched_inputs = request.get_inputs()

                # Inference (profile memory)
                if self.mode == "full" or self.mode == "memory":
                    with record.profile_memory("forward") as prof:
                        predict(**batched_inputs)
                    tree.add(f"[bold green]✓[/bold green] {prof}).")

                # Inference Warmup
                if self.mode == "full" or self.mode == "execution":
                    with record.profile_execution("forward_warmup", duration=2) as prof:
                        [predict(**batched_inputs) for _ in prof.iterator]
                    tree.add(f"[bold green]✓[/bold green] {prof}).")

                # Inference (profile execution)
                if self.mode == "full" or self.mode == "execution":
                    with record.profile_execution("forward", duration=5) as prof:
                        [predict(**batched_inputs) for _ in prof.iterator]
                    tree.add(f"[bold green]✓[/bold green] {prof}).")

            except Exception as e:
                logger.error(f"Failed to profile, e={e}")
                raise e

            finally:
                # Destroy
                with record.profile_memory("cleanup") as prof:
                    try:
                        del model.model
                    except Exception:
                        pass
                    model.model = None
                    gc.collect()
                    torch.cuda.empty_cache()
                tree.add(f"[bold green]✓[/bold green] {prof}).")

        # Update the record with more metadata
        # key metrics: (prof.forward::execution.*_utilization, prof.forward::memory_*::allocated, prof.wrap::memory_*::allocated)
        for key, value in request.kwargs.items():
            record.update(key, value)
        record.update("forward::memory_gpu::allocated", record.prof["forward::memory_gpu::post"])
        record.update("forward::memory_cpu::allocated", record.prof["forward::memory_cpu::post"])
        print(tree)

    def run(self) -> None:
        """Run all benchmarks."""
        failed = {}
        st_t = time.time()

        print()
        print(f"[white]{self}[/white]")
        with Profiler() as self.prof, torch.inference_mode():
            for _idx, request in enumerate(self.requests):
                # Skip subsequent benchmarks with same name if previous runs failed
                # Note: This is to avoid running benchmarks that previously failed
                # due to OOM with smaller batch sizes.
                if request.model_id in failed:
                    logger.debug(f"Skipping benchmark, since previous run failed: {request}")
                    continue
                try:
                    self._benchmark(request)
                except Exception as exc:
                    logger.error(f"Profiling failed: {request}, e={exc}.")
                    failed[request.model_id] = request
                    continue
        print(f"[bold green] ✅ Benchmarks completed (elapsed={time.time() - st_t:.1f}s) [/bold green]")

    def save(self) -> str:
        """Save profiled results to JSON."""
        from nos.constants import NOS_PROFILE_CATALOG_PATH

        NOS_PROFILE_DIR = NOS_CACHE_DIR / "profile"
        NOS_PROFILE_DIR.mkdir(parents=True, exist_ok=True)

        version_str = __version__.replace(".", "-")
        date_str = datetime.datetime.utcnow().strftime("%Y%m%d")
        profile_path = Path(NOS_PROFILE_DIR) / f"nos-profile--{version_str}--{date_str}--{self.device_name}.json"
        print(
            f"[bold green] Writing profile results to {profile_path} (records={len(self.prof.records)})[/bold green]"
        )
        self.prof.save(profile_path)

        # Copy the profile to the metadata catalog
        shutil.copyfile(str(profile_path), str(NOS_PROFILE_CATALOG_PATH))

        return str(profile_path)
