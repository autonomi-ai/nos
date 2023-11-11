from typing import Iterator, Tuple

import numpy as np
import rich.console
import rich.status
import typer
from PIL import Image

from nos import hub
from nos.common.profiler import Profiler
from nos.common.spec import ModelSpec
from nos.common.tasks import TaskType
from nos.logging import logger
from nos.test.utils import NOS_TEST_IMAGE


profile_cli = typer.Typer(name="profile", help="NOS Profiler CLI.", no_args_is_help=True)
console = rich.console.Console()


# def profile_model(model_id: str, device_id: int = 0, verbose: bool = False) -> Profiler:
#     """Main entrypoint for profiling a model."""
#     import torch

#     from nos.common.profiler import ModelProfiler, ModelProfileRequest
#     from nos.common.spec import ModelSpec, ModelSpecMetadata
#     from nos.server._runtime import InferenceServiceRuntime

#     # TODO (spillai): Pytorch cuda.devices are reported in the descending order of memory,
#     # so we need to force them to match the `nvidia-smi` order. Setting CUDA_DEVICE_ORDER=PCI_BUS_ID
#     # allows us to keep the order consistent with `nvidia-smi`.
#     assert torch.cuda.device_count() > 0, "No CUDA devices found, profiling is only supported on NVIDIA currently."
#     # assert os.getenv("CUDA_DEVICE_ORDER", "") == "PCI_BUS_ID", "CUDA_DEVICE_ORDER must be PCI_BUS_ID."

#     # Load model spec
#     try:
#         spec: ModelSpec = hub.load_spec(model_id)
#     except Exception as e:
#         logger.error(f"Failed to load model spec for {model_id}: {e}")
#         raise typer.Exit(1)

#     runtime = InferenceServiceRuntime.detect()
#     logger.info(f"Detected runtime: {runtime}")

#     # Get the device information (nvidia-gpu model type from torch)
#     device: str = torch.cuda.get_device_properties(device_id).name.lower().replace(" ", "-")
#     logger.info(f"Using device: {device}")

#     # # Update the model spec metadata
#     # metadata = ModelSpecMetadata(
#     #     id=spec.id,
#     #     method=spec.default_method,
#     #     task=spec.task(spec.default_method),
#     #     resources=ModelResources(runtime=runtime, device=device, device_memory=2 * 1024**3),
#     # )
#     # spec.set_metadata(spec.default_method, metadata)

#     # Load test image
#     pil_im = Image.open(NOS_TEST_IMAGE)

#     # Profile the model
#     profiler = ModelProfiler(mode="full", device_id=device_id)
#     profiler.add(
#         ModelProfileRequest(
#             model_id=spec.id,
#             method=spec.default_method,
#             get_inputs=lambda : _model_inputs(spec.id, "{
#                 "images": [np.asarray(pil_im.resize((224, 224))) for _ in range(2)]
#             },
#             batch_size=1,
#             shape=(224, 224),
#         )
#     )
#     profiler.add(
#         ModelProfileRequest(
#             model_id=spec.id,
#             method=spec.default_method,
#             get_inputs=lambda: {
#                 "images": [np.asarray(pil_im.resize((224, 224))) for _ in range(2)]
#             },
#             batch_size=2,
#             shape=(224, 224),
#         )
#     )
#     profiler.run()
#     profiler.save()
#     return profiler


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
    else:
        raise ValueError(f"Unsupported task: {task}")


def _model_methods() -> Iterator:
    models = hub.list()
    for model_id in models:
        spec: ModelSpec = hub.load_spec(model_id)
        for method in spec.signature:
            yield model_id, method, spec


def profile_all_models(device_id: int = 0, verbose: bool = False) -> Profiler:
    """Main entrypoint for profiling all models."""
    import torch

    from nos.common.profiler import ModelProfiler, ModelProfileRequest
    from nos.server._runtime import InferenceServiceRuntime

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
    profiler = ModelProfiler(mode="full", device_id=device_id)

    for model_id, method, spec in _model_methods():
        task: TaskType = spec.task(method)
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
        else:
            logger.warning(f"Unsupported task: {task}, skipping.")
            continue

    profiler.run()
    profiler.save()
    return profiler


@profile_cli.command(name="model")
def _profile_model(
    ctx: typer.Context,
    model_id: str = typer.Option(..., "-m", "--model-id", help="Model identifier."),
    device_id: int = typer.Option(0, "--device-id", "-d", help="Device ID to use."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose profiling."),
):
    """Profile a specific model by its identifier.

    Note:
        If you have multiple GPUs and want to profile on a specific one, while
        keeping the correct PCI BUS id, you can use the following command:

        $ CUDA_DEVICE_ORDER=PCI_BUS_ID nos profile -m <model-id> -d 0 --verbose
    """
    # profile_model(model_id, device_id=device_id, verbose=verbose)
    pass


@profile_cli.command(name="all")
def _profile_all_models(
    ctx: typer.Context,
    device_id: int = typer.Option(0, "--device-id", "-d", help="Device ID to use."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose profiling."),
):
    """Profile all models.

    Note:
        If you have multiple GPUs and want to profile on a specific one, while
        keeping the correct PCI BUS id, you can use the following command:

        $ CUDA_DEVICE_ORDER=PCI_BUS_ID nos profile -m <model-id> -d 0 --verbose
    """
    # profile_model(model_id, device_id=device_id, verbose=verbose)
    profile_all_models(device_id=device_id, verbose=verbose)


@profile_cli.command(name="list")
def _profile_list():
    from typing import List

    from rich.table import Table

    from nos import hub

    models: List[str] = hub.list()

    table = Table(title="Models")
    table.add_column("Model ID", style="cyan")
    table.add_column("Method", style="magenta")
    table.add_column("Task", style="green")
    table.add_column("Runtime", style="yellow")
    table.add_column("Device", style="yellow")
    table.add_column("Device Memory", style="yellow")

    for model_id in models:
        spec: ModelSpec = hub.load_spec(model_id)
        table.add_row(
            model_id,
            spec.method,
            spec.task,
            spec.resources.runtime,
            spec.resources.device,
            f"{spec.resources.device_memory / 1024**3:.2f} GB",
        )
    console.print(table)
