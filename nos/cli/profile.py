"""Profiler CLI for NOS.

Note:
    If you have multiple GPUs and want to profile on a specific one, while
    keeping the correct PCI BUS id, you can use the following command:

    $ CUDA_DEVICE_ORDER=PCI_BUS_ID nos profile -m <model-id> -d 0 --verbose
"""

from typing import Iterator, Tuple

import humanize
import numpy as np
import typer
from PIL import Image
from rich import print

from nos import hub
from nos.common.profiler import ModelProfiler, ModelProfileRequest, Profiler
from nos.common.spec import ModelSpec
from nos.common.tasks import TaskType
from nos.logging import logger
from nos.server._runtime import InferenceServiceRuntime
from nos.test.utils import NOS_TEST_IMAGE


profile_cli = typer.Typer(name="profile", help="NOS Profiler CLI.", no_args_is_help=True)


def _model_inputs(task: TaskType, shape: Tuple[int, int] = None, batch_size: int = 1):
    if task == TaskType.IMAGE_EMBEDDING:
        im = Image.open(NOS_TEST_IMAGE)
        return {"images": [np.asarray(im.resize(shape)) for _ in range(batch_size)]}
    elif task == TaskType.TEXT_EMBEDDING:
        return {"texts": ["A photo of a cat."] * batch_size}
    elif task == TaskType.OBJECT_DETECTION_2D:
        im = Image.open(NOS_TEST_IMAGE)
        return {"images": [np.asarray(im.resize(shape)) for _ in range(batch_size)]}
    elif task == TaskType.IMAGE_GENERATION:
        assert batch_size == 1, "Image generation only supports batch_size=1 currently."
        return {"prompts": ["A photo of a cat."], "num_images": 1, "num_inference_steps": 10}
    elif task == TaskType.AUDIO_TRANSCRIPTION:
        from nos.test.utils import NOS_TEST_AUDIO

        assert batch_size == 1, "Audio transcription only supports batch_size=1 currently."
        return {"path": NOS_TEST_AUDIO, "chunk_length_s": 30, "return_timestamps": True}
    elif task == TaskType.DEPTH_ESTIMATION_2D:
        im = Image.open(NOS_TEST_IMAGE)
        return {"images": [np.asarray(im.resize(shape)) for _ in range(batch_size)]}
    elif task == TaskType.IMAGE_SEGMENTATION_2D:
        im = Image.open(NOS_TEST_IMAGE)
        return {"images": [np.asarray(im.resize(shape)) for _ in range(batch_size)]}
    elif task == TaskType.IMAGE_SUPER_RESOLUTION:
        im = Image.open(NOS_TEST_IMAGE)
        return {"images": [np.asarray(im.resize(shape)) for _ in range(batch_size)]}
    else:
        raise ValueError(f"Unsupported task: {task}")


def _model_methods(model_id: str = None) -> Iterator[Tuple[str, str, ModelSpec]]:
    models = hub.list()
    for _model_id in models:
        if model_id is not None and model_id != _model_id:
            continue
        spec: ModelSpec = hub.load_spec(_model_id)
        for method in spec.signature:
            yield _model_id, method, spec


def profile_models(model_id: str = None, device_id: int = 0, save: bool = False, verbose: bool = False) -> Profiler:
    """Main entrypoint for profiling all models."""
    import torch

    # TODO (spillai): Pytorch cuda.devices are reported in the descending order of memory,
    # so we need to force them to match the `nvidia-smi` order. Setting CUDA_DEVICE_ORDER=PCI_BUS_ID
    # allows us to keep the order consistent with `nvidia-smi`.
    assert torch.cuda.device_count() > 0, "No CUDA devices found, profiling is only supported on NVIDIA currently."
    # assert os.getenv("CUDA_DEVICE_ORDER", "") == "PCI_BUS_ID", "CUDA_DEVICE_ORDER must be PCI_BUS_ID."

    runtime = InferenceServiceRuntime.detect()
    logger.info(f"Detected runtime: {runtime}")

    # Get the device information (nvidia-gpu model type from torch)
    device: str = torch.cuda.get_device_properties(device_id).name.lower().replace(" ", "-")
    logger.info(f"Using device: {device}")

    # Profile all models
    profiler = ModelProfiler(mode="full", runtime=runtime, device_id=device_id)
    for model_id, method, spec in _model_methods(model_id):  # noqa: B020
        logger.debug(f"Profiling model: {model_id} (method: {method})")
        if model_id is None and model_id != model_id:
            logger.debug(f"Skipping model: {model_id} (not requested).")
            continue

        task: TaskType = spec.task(method)
        logger.debug(f"Task: {task}")
        if task == TaskType.IMAGE_EMBEDDING:
            profiler.add(
                ModelProfileRequest(
                    model_id=model_id,
                    method=method,
                    get_inputs=lambda: _model_inputs(task=TaskType.IMAGE_EMBEDDING, shape=(224, 224), batch_size=1),
                    batch_size=1,
                    shape=(224, 224),
                ),
            )
        elif task == TaskType.TEXT_EMBEDDING:
            profiler.add(
                ModelProfileRequest(
                    model_id=model_id,
                    method=method,
                    get_inputs=lambda: _model_inputs(task=TaskType.TEXT_EMBEDDING, batch_size=1),
                    batch_size=1,
                    shape=None,
                ),
            )
        elif task == TaskType.OBJECT_DETECTION_2D:
            profiler.add(
                ModelProfileRequest(
                    model_id=model_id,
                    method=method,
                    get_inputs=lambda: _model_inputs(
                        task=TaskType.OBJECT_DETECTION_2D, shape=(640, 480), batch_size=1
                    ),
                    batch_size=1,
                    shape=(640, 480),
                ),
            )
        elif task == TaskType.IMAGE_GENERATION:
            profiler.add(
                ModelProfileRequest(
                    model_id=model_id,
                    method=method,
                    get_inputs=lambda: _model_inputs(task=TaskType.IMAGE_GENERATION, batch_size=1),
                    batch_size=1,
                    shape=None,
                ),
            )
        elif task == TaskType.AUDIO_TRANSCRIPTION:
            profiler.add(
                ModelProfileRequest(
                    model_id=model_id,
                    method=method,
                    get_inputs=lambda: _model_inputs(task=TaskType.AUDIO_TRANSCRIPTION, batch_size=1),
                    batch_size=1,
                    shape=None,
                ),
            )
        elif task == TaskType.DEPTH_ESTIMATION_2D:
            profiler.add(
                ModelProfileRequest(
                    model_id=model_id,
                    method=method,
                    get_inputs=lambda: _model_inputs(
                        task=TaskType.DEPTH_ESTIMATION_2D, shape=(640, 480), batch_size=1
                    ),
                    batch_size=1,
                    shape=(640, 480),
                ),
            )
        elif task == TaskType.IMAGE_SEGMENTATION_2D:
            profiler.add(
                ModelProfileRequest(
                    model_id=model_id,
                    method=method,
                    get_inputs=lambda: _model_inputs(
                        task=TaskType.IMAGE_SEGMENTATION_2D, shape=(640, 480), batch_size=1
                    ),
                    batch_size=1,
                    shape=(640, 480),
                ),
            )
        elif task == TaskType.IMAGE_SUPER_RESOLUTION:
            profiler.add(
                ModelProfileRequest(
                    model_id=model_id,
                    method=method,
                    get_inputs=lambda: _model_inputs(
                        task=TaskType.IMAGE_SUPER_RESOLUTION, shape=(160, 120), batch_size=1
                    ),
                    batch_size=1,
                    shape=(160, 120),
                ),
            )
        else:
            logger.warning(f"Unsupported task: {task}, skipping.")
            continue

    # Run the profiler, and optionally save the catalog
    profiler.run()
    if save:
        profiler.save()
    return profiler


@profile_cli.command(name="model")
def _profile_model(
    model_id: str = typer.Option(..., "-m", "--model-id", help="Model identifier."),
    device_id: int = typer.Option(0, "--device-id", "-d", help="Device ID to use."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose profiling."),
):
    """Profile a specific model by its identifier."""
    profile_models(model_id, device_id=device_id, save=True, verbose=verbose)


@profile_cli.command(name="all")
def _profile_all_models(
    device_id: int = typer.Option(0, "--device-id", "-d", help="Device ID to use."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose profiling."),
):
    """Profile all models."""
    profile_models(device_id=device_id, verbose=verbose)


@profile_cli.command(name="rebuild-catalog")
def _profile_rebuild_catalog(
    device_id: int = typer.Option(0, "--device-id", "-d", help="Device ID to use."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose profiling."),
):
    """Profile all models and save the catalog."""
    profile_models(device_id=device_id, save=True, verbose=verbose)


@profile_cli.command(name="list")
def _profile_list():
    """List all models and their methods."""
    from rich.table import Table

    from nos import hub

    table = Table(title="[green]  Models [/green]")
    table.add_column("model_id", max_width=30)
    table.add_column("method")
    table.add_column("task")
    table.add_column("runtime")
    table.add_column("device")
    table.add_column("it/s")
    table.add_column("cpu_memory")
    table.add_column("cpu_util")
    table.add_column("gpu_memory")
    table.add_column("gpu_util")

    for model in hub.list(private=False):
        spec: ModelSpec = hub.load_spec(model)
        for method in spec.signature:
            metadata = spec.metadata(method)
            try:
                if hasattr(metadata, "resources"):
                    runtime = metadata.resources.runtime
                    device = "-".join(metadata.resources.device.split("-")[-2:])
                    cpu_memory = f"{humanize.naturalsize(metadata.resources.memory, binary=True)}"
                    gpu_memory = f"{humanize.naturalsize(metadata.resources.device_memory, binary=True)}"
                if hasattr(metadata, "profile"):
                    it_s = f'{metadata.profile["prof.forward::execution.num_iterations"] * 1e3 / metadata.profile["prof.forward::execution.total_ms"]:.1f}'
                    cpu_util = f'{metadata.profile["prof.forward::execution.cpu_utilization"]:0.2f}'
                    gpu_util = f'{metadata.profile["prof.forward::execution.gpu_utilization"]:0.2f}'
                else:
                    print("no metadata")
            except Exception:
                it_s = "-"
                cpu_util = "-"
                gpu_util = "-"
                cpu_memory = "-"
                gpu_memory = "-"
                runtime, device = None, None
            table.add_row(
                f"[green]{model}[/green]",
                method,
                spec.task(method),
                runtime,
                device,
                it_s,
                cpu_memory,
                cpu_util,
                gpu_memory,
                gpu_util,
            )
    print(table)
