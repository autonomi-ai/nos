import gc
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import torch
import typer
from PIL import Image
from rich.console import Console

from nos import hub
from nos.common import TaskType, tqdm
from nos.common.profiler import Profiler
from nos.common.system import get_system_info, has_gpu, is_inside_docker
from nos.constants import NOS_CACHE_DIR
from nos.logging import logger
from nos.test.utils import NOS_TEST_IMAGE
from nos.version import __version__


@dataclass(frozen=True)
class ProfileResult:
    profile: pd.DataFrame
    """Profiled results as a dataframe."""
    acc_info: Dict[str, Any] = None
    """Accelerator (GPU) info."""

    def __post_init__(self):
        assert isinstance(self.profile, pd.DataFrame), "Profile must be a DataFrame."
        assert self.profile.index.name == "key", "Profile must have a 'key' index."
        assert "device_id" in self.profile.columns, "Profile must have a device_id."

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary."""
        return {
            "profile": self.profile.to_dict(),
            "sys_info": get_system_info(docker=False, gpu=True),
        }

    def save(self, filename: Union[str, Path]) -> None:
        """Save profiled results to JSON."""
        import json

        with open(str(filename), "w") as f:
            json.dump(self.to_dict(), f, indent=4)


benchmark_cli = typer.Typer(name="benchmark", help="NOS Benchmark CLI.", no_args_is_help=True)
console = Console()


@benchmark_cli.command("list", help="List model benchmark results.")
def _benchmark_list() -> None:
    pass


@benchmark_cli.command("run", help="Run model benchmarks, and profile if needed.")
def _benchmark_run() -> None:
    pass


@dataclass(frozen=True)
class BenchmarkModel:
    task: TaskType
    """Task type."""
    name: str
    """Model name."""
    batch_size: int
    """Batch size."""
    shape: Tuple[int]
    """Input shape."""
    image: Image.Image
    """Input image."""
    get_inputs: Callable
    """Callable to get inputs."""

    def __repr__(self) -> str:
        return f"BenchmarkModel (task={self.task}, name={self.name}, bsize={self.batch_size}, shape={self.shape}, image={self.image.size})"


@dataclass
class BenchmarkProfiler:
    """Benchmark profiler.

    Usage:
        >>> from nos.cli.benchmark import BenchmarkProfiler
        >>> profiler = BenchmarkProfiler(models=[...])
    """

    models: List[BenchmarkModel] = field(default_factory=list)
    """Models to benchmark."""
    prof: Profiler = None
    """Profiler used for benchmarking."""
    device_id: int = -1
    """Device ID."""
    device_name: str = None
    """Device name."""
    device: torch.device = None
    """Torch Device to run benchmark."""

    def __repr__(self) -> str:
        repr_str = (
            f"BenchmarkProfiler (models={len(self.models)}, device_name={self.device_name}, device={self.device})"
        )
        for model in self.models:
            repr_str += f"\n\t{model}"
        return repr_str

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
            unique_gpu_devices = gpu_devices_df["device_name"].unique()
            console.print(f"Found GPU devices: {len(gpu_devices_df)}, unique: {len(unique_gpu_devices)}")
            if len(unique_gpu_devices) > 1:
                console.print(f"Multiple devices detected, selecting {unique_gpu_devices[0]}.")
            console.print(gpu_devices_df.to_markdown())

            # Select the appropriate device, and retrieve its name
            self.device_name = gpu_devices_df["device_name"].iloc[self.device_id]
            self.device_name = self.device_name.replace(" ", "-").lower()
            self.device = torch.device(f"cuda:{self.device_id}")
            assert self.device.index == self.device_id, "Device index mismatch."
            assert self.device.type == "cuda", "Device type mismatch."
        else:
            # CPU device
            self.device_id, self.device_name = -1, "cpu"
            self.device = torch.device("cpu")

    def add(self, model: BenchmarkModel) -> None:
        """Add a model to the profiler."""
        self.models.append(model)

    def _benchmark(self, bmodel: BenchmarkModel) -> None:
        """Benchmark / profile a specific model."""

        # Add a new record to profile
        record = self.prof.add(
            f"nos::{bmodel.name}",
            device_name=self.device_name,
            device_type=self.device.type,
            device_index=self.device.index,
            batch_size=bmodel.batch_size,
            shape=bmodel.shape,
        )
        with record.profile_memory("wrap"):
            try:
                # Initialize (profile memory)
                with record.profile_memory("init"):
                    spec = hub.load_spec(bmodel.name, bmodel.task)
                    model = hub.load(spec.name, spec.task)
                    predict = getattr(model, spec.signature.method_name)

                # Inference (profile memory)
                batched_inputs = bmodel.get_inputs(bmodel.image, bmodel.shape, bmodel.batch_size)
                with record.profile_memory("forward"):
                    predict(**batched_inputs)

                # Inference Warmup
                with record.profile_execution("forward_warmup", duration=5) as p:
                    [predict(**batched_inputs) for _ in p.iterator]

                # Inference (profile execution)
                with record.profile_execution("forward", duration=10) as p:
                    [predict(**batched_inputs) for _ in p.iterator]

            except Exception as e:
                logger.error(f"Failed to profile, e={e}")
                raise e

            finally:
                # Destroy
                with record.profile_memory("cleanup"):
                    try:
                        del model.model
                    except Exception:
                        pass
                    model.model = None
                    gc.collect()
                    torch.cuda.empty_cache()

    def run(self) -> None:
        """Run all benchmarks."""
        failed = {}
        st_t = time.time()

        console.print("[bold green] Running benchmarks ...")
        console.print(f"[bold white] {self} [/bold white]")
        with Profiler() as self.prof, torch.inference_mode():
            pbar = tqdm(self.models)
            for bmodel in pbar:
                # Skip subsequent benchmarks with same name if previous runs failed
                # Note: This is to avoid running benchmarks that previously failed
                # due to OOM with smaller batch sizes.
                if bmodel.name in failed and bmodel.batch_size >= failed[bmodel.name].batch_size:
                    logger.debug(f"Skipping benchmark, since previous run failed: {bmodel}")
                    continue
                pbar.set_description(
                    f"Profiling [name={bmodel.name}, bsize={bmodel.batch_size}, shape={bmodel.shape}, device={self.device_name}]"
                )
                try:
                    self._benchmark(bmodel)
                except Exception:
                    logger.error(f"Profiling failed: {bmodel}.")
                    failed[bmodel.name] = bmodel
                    continue
        console.print(f"[bold green] Benchmarks completed (elapsed={time.time() - st_t:.1f}s) [/bold green]")

    def save(self) -> None:
        """Save profiled results to JSON."""
        NOS_PROFILE_DIR = NOS_CACHE_DIR / "profile"
        NOS_PROFILE_DIR.mkdir(parents=True, exist_ok=True)

        version_str = __version__.replace(".", "-")
        date_str = datetime.utcnow().strftime("%Y%m%d")
        profile_path = Path(NOS_PROFILE_DIR) / f"nos-profile--{version_str}--{date_str}--{self.device_name}.json"
        console.print(
            f"[bold green] Writing profile results to {profile_path} (records={len(self.prof.records)})[/bold green]"
        )
        self.prof.save(profile_path)


@benchmark_cli.command("profile", help="Profile all models.")
def _benchmark_profile(
    device_id: int = typer.Option(0, "--device-id", "-d", help="Device ID to use."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose profiling."),
) -> None:
    """Profile all models.

    Usage:
        $ CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 nos benchmark profile --verbose
    """

    if not is_inside_docker():
        logger.warning("nos benchmarks should be ideally run within docker, continuing ...")

    # TODO (spillai): Pytorch cuda.devices are reported in the descending order of memory,
    # so we need to force them to match the `nvidia-smi` order. Setting CUDA_DEVICE_ORDER=PCI_BUS_ID
    # allows us to keep the order consistent with `nvidia-smi`.
    assert os.getenv("CUDA_DEVICE_ORDER", "") == "PCI_BUS_ID", "CUDA_DEVICE_ORDER must be PCI_BUS_ID."

    # Disables SettingWithCopyWarning
    pd.set_option("mode.chained_assignment", None)

    # Load test image
    pil_im = Image.open(NOS_TEST_IMAGE)

    # Create benchmark experiments from varied tasks, batch sizes and input shapes
    BATCH_SIZES = [2**b for b in range(11)]

    benchmark = BenchmarkProfiler(device_id=device_id)
    for spec in hub.list():
        if spec.task == TaskType.IMAGE_EMBEDDING:
            SHAPES = [(224, 224), (640, 480)]
            for (batch_size, shape) in product(BATCH_SIZES, SHAPES):
                benchmark.add(
                    BenchmarkModel(
                        task=spec.task,
                        name=spec.name,
                        batch_size=batch_size,
                        shape=shape,
                        image=pil_im,
                        get_inputs=lambda im, shape, batch_size: {
                            "images": [np.asarray(im.resize(shape)) for _ in range(batch_size)]
                        },
                    )
                )
        elif spec.task == TaskType.OBJECT_DETECTION_2D:
            SHAPES = [(640, 480), (1280, 960), (1920, 1440)]
            for (batch_size, shape) in product(BATCH_SIZES, SHAPES):
                benchmark.add(
                    BenchmarkModel(
                        task=spec.task,
                        name=spec.name,
                        batch_size=batch_size,
                        shape=shape,
                        image=pil_im,
                        get_inputs=lambda im, shape, batch_size: {
                            "images": [np.asarray(im.resize(shape)) for _ in range(batch_size)]
                        },
                    )
                )

    # Run benchmarks
    benchmark.models = benchmark.models
    benchmark.run()
    benchmark.save()
