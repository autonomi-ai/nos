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
            # check if CUDA_VISIBLE_DEVICES is set and default to 0:
            if os.getenv("CUDA_VISIBLE_DEVICES", None) is None:
                os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
        self.profiling_data = {}  # The actual metadata collected inside this profiling record.

    @contextlib.contextmanager
    def profile_execution(self, name: str = None, iterations: int = None, duration: float = None) -> profile_execution:
        """Context manager for profiling execution time."""
        with profile_execution(f"{name}", iterations=iterations, duration=duration) as prof_ctx_mgr:
            yield prof_ctx_mgr

        # Update the profiing data with that collected by the context manager during model execution.
        if prof_ctx_mgr.name not in self.profiling_data:
            self.profiling_data[prof_ctx_mgr.name] = prof_ctx_mgr.execution_stats.__dict__
        else:
            self.profiling_data[prof_ctx_mgr.name].update(prof_ctx_mgr.execution_stats.__dict__)

    @contextlib.contextmanager
    def profile_memory(self, name: str = None) -> profile_memory:
        """Context manager for profiling memory usage."""
        with profile_memory(f"{name}") as prof_ctx_mgr:
            yield prof_ctx_mgr
        # TODO (spillai): This is to avoid nested namespaces in the profile data dict.
        self.profiling_data.update(prof_ctx_mgr.memory_usage())

    def update(self, key: str, value: Any) -> None:
        """Helper function to place any updates to profiling results in the data dict."""
        self.profiling_data[key] = value

    def as_dict(self) -> Dict[str, Any]:
        """Return a dictionary representation of the profiler record."""
        return {
            "namespace": self.namespace,
            "profiling_data": self.profiling_data,
            **self.kwargs,
        }


@dataclass
class Profiler:
    """NOS profiler as a context manager."""

    records: List[profiler_record] = field(default_factory=list)
    """List of profiler records. These will be added to the catalog then
       populated as a context manager when running the model. They map to
       methods on a particular model, and we try not to duplicate them for
       a single method (i.e. you can add multiple entries with the same
       namespace for e.g. CLIP, but when dumping the profiling results
       we only retrieve the first hit for each function signature).

       TODO: Prevent profiler from storing multiple records for the same
       function signature.
    """

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
        """Add a profiler record so we can record results for model pass."""
        self.records.append(profiler_record(namespace, **kwargs))
        return self.records[-1]

    # We store everything as json but manipulate as df, need conversions for these.
    def as_df(self) -> pd.DataFrame:
        """Return a dataframe representation of the profiled result."""
        metadata = ProfilerMetadata()
        df = pd.DataFrame([r.as_dict() for r in self.records]).assign(date=metadata.date, version=metadata.version)
        return df

    @classmethod
    def from_df(cls, df: pd.DataFrame) -> "Profiler":
        """Load profiled results from dataframe."""
        records = []
        for _idx, row in df.iterrows():
            namespace = row["namespace"]
            kwargs = row.drop("namespace").to_dict()
            record = profiler_record(namespace, **kwargs)
            for key, value in row.items():
                record.update(key, value)
            records.append(record)

        return cls(records)

    @classmethod
    def from_json_path(cls, filename: Union[Path, str]) -> "Profiler":
        """Load profiled results from JSON."""
        # if the filename doesn't exist, initialize an empty profiler:
        if not Path(filename).exists():
            return cls()
        else:
            df = pd.read_json(str(filename), orient="records")
            return cls.from_df(df)

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
    profiler: Profiler = None
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

        # Check if we have a previous catalog and load it
        from nos.constants import NOS_PROFILE_CATALOG_PATH

        logger.debug("NOS profile catalog path: %s", NOS_PROFILE_CATALOG_PATH)

        if NOS_PROFILE_CATALOG_PATH.exists():
            self.profiler = Profiler()
            self.profiler.from_json_path(NOS_PROFILE_CATALOG_PATH)
        else:
            logger.debug(f"Profile catalog not found (filename={NOS_PROFILE_CATALOG_PATH}).")

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

        tree = Tree(
            f"🔥 [bold white]Profiling (name={request.model_id}, device={self.device_name}, kwargs={request.kwargs})[/bold white]"
        )

        # Needs to be set for utilization stats to work
        assert os.getenv("CUDA_VISIBLE_DEVICES", None) is not None, "CUDA_VISIBLE_DEVICES is not set."

        # We need to create a profiling record for this request, or udpate an existing one.
        # Check if we already have a record for this request in the catalog and remove it:
        record = None
        for existing_record in self.profiling_data.records:
            if existing_record.namespace == f"nos::{request.model_id}":
                record = existing_record
                logger.debug("Found existing record for %s, updating it.", request.model_id)
                break

        if record is None:
            # Otherwise create a new one
            logger.debug("Creating new record for %s")
            record = self.profiling_data.add(
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
                with record.profile_memory("init") as prof_ctx_mgr:
                    model = hub.load(request.model_id)
                    predict = getattr(model, request.method)
                tree.add(f"[bold green]✓[/bold green] {prof_ctx_mgr}).")

                batched_inputs = request.get_inputs()

                # Inference (profile memory)
                if self.mode == "full" or self.mode == "memory":
                    with record.profile_memory("forward") as prof_ctx_mgr:
                        predict(**batched_inputs)
                    tree.add(f"[bold green]✓[/bold green] {prof_ctx_mgr}).")

                # Inference Warmup
                if self.mode == "full" or self.mode == "execution":
                    with record.profile_execution("forward_warmup", duration=2) as prof_ctx_mgr:
                        [predict(**batched_inputs) for _ in prof_ctx_mgr.iterator]
                    tree.add(f"[bold green]✓[/bold green] {prof_ctx_mgr}).")

                # Inference (profile execution)
                if self.mode == "full" or self.mode == "execution":
                    with record.profile_execution("forward", duration=5) as prof_ctx_mgr:
                        [predict(**batched_inputs) for _ in prof_ctx_mgr.iterator]
                    tree.add(f"[bold green]✓[/bold green] {prof_ctx_mgr}).")

            except Exception as e:
                logger.error(f"Failed to profile, e={e}")
                raise e

            finally:
                # Destroy
                with record.profile_memory("cleanup") as prof_ctx_mgr:
                    try:
                        del model.model
                    except Exception:
                        pass
                    model.model = None
                    gc.collect()
                    torch.cuda.empty_cache()
                tree.add(f"[bold green]✓[/bold green] {prof_ctx_mgr}).")

        # Update the record with more metadata
        # key metrics: (prof.forward::execution.*_utilization, prof.forward::memory_*::allocated, prof.wrap::memory_*::allocated)
        for key, value in request.kwargs.items():
            record.update(key, value)
        record.update("forward::memory_gpu::allocated", record.profiling_data["forward::memory_gpu::post"])
        record.update("forward::memory_cpu::allocated", record.profiling_data["forward::memory_cpu::post"])
        record.update(
            "forward::execution.gpu_utilization", record.profiling_data["forward::execution"]["gpu_utilization"]
        )
        record.update(
            "forward::execution.cpu_utilization", record.profiling_data["forward::execution"]["cpu_utilization"]
        )
        print(tree)

    def run(self) -> None:
        """Run all benchmarks."""
        failed = {}
        st_t = time.time()

        print(f"[white]{self}[/white]")
        from nos.constants import NOS_PROFILE_CATALOG_PATH

        self.profiling_data = Profiler.from_json_path(NOS_PROFILE_CATALOG_PATH)
        with torch.inference_mode():
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

    def save(self, catalog_path: str = None) -> str:
        """Save profiled results to JSON."""
        NOS_PROFILE_DIR = NOS_CACHE_DIR / "profile"
        NOS_PROFILE_DIR.mkdir(parents=True, exist_ok=True)

        version_str = __version__.replace(".", "-")
        date_str = datetime.datetime.utcnow().strftime("%Y%m%d")
        profile_path = Path(NOS_PROFILE_DIR) / f"nos-profile--{version_str}--{date_str}--{self.device_name}.json"
        print(
            f"[bold green] Writing profile results to {profile_path} (records={len(self.profiling_data.records)})[/bold green]"
        )
        self.profiling_data.save(profile_path)

        # Still need to copy to the default catalog path for now
        from nos.constants import NOS_PROFILE_CATALOG_PATH

        shutil.copyfile(str(profile_path), str(NOS_PROFILE_CATALOG_PATH))

        # This is a WIP to allow us to map the profiling catalog to a
        # filemount when running in GCP etc.
        if catalog_path is not None:
            # Copy the profile to the metadata catalog
            Path(catalog_path).mkdir(parents=True, exist_ok=True)
            full_catalog_path = Path(catalog_path) / profile_path.name
            shutil.copyfile(str(profile_path), str(Path(full_catalog_path)))

        return str(profile_path)
