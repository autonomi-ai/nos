"""gRPC-based Client CLI for NOS.

Usage:
    $ nos serve-grpc --help
    $ nos serve-grpc list
    $ nos serve-grpc deploy -m stabilityai/stable-diffusion-v2
    $ nos serve-grpc img2vec -i tests/test_data/test.jpg
    $ nos serve-grpc txt2vec -i 'Hello World!'
    $ nos serve-grpc txt2img -i 'a cat dancing on the grass'

"""
import time
from dataclasses import dataclass

import grpc
import ray
import rich.console
import rich.status
import rich.table
import typer
from google.protobuf import empty_pb2
from tqdm import tqdm

from nos.cli.utils import AsyncTyper
from nos.experimental.grpc import import_module


nos_service_pb2 = import_module("nos_service_pb2")
nos_service_pb2_grpc = import_module("nos_service_pb2_grpc")

serve_grpc_cli = AsyncTyper(name="serve-grpc", help="NOS gRPC Serve CLI.", no_args_is_help=True)
console = rich.console.Console()


@dataclass
class gRPCConfig:
    """Common gRPC options"""

    address: str


@serve_grpc_cli.callback()
def grpc_config(
    ctx: typer.Context,
    address: str = typer.Option("localhost:50051", "-a", "--address", help="Address of the gRPC server."),
):
    """Common gRPC options"""
    ctx.obj = gRPCConfig(address)
    # TODO (spillai): Deploy the gRPC server here in the background (as a docker daemon)
    # TOOD (spillai): Ping the gRPC server otherwise raise an error


@serve_grpc_cli.async_command("list", help="List all gRPC deployments.")
async def _grpc_serve_list(ctx: typer.Context):
    """List all gRPC deployments."""

    async with grpc.aio.insecure_channel(ctx.obj.address) as channel:
        stub = nos_service_pb2_grpc.InferenceServiceStub(channel)
        try:
            response: nos_service_pb2.ModelListResponse = await stub.ListModels(empty_pb2.Empty())
            console.print(response.models)
        except grpc.RpcError as e:
            console.print(f"[red] ✗ Failed to list models ({e}).[/red]")


@serve_grpc_cli.async_command("deploy", help="Create a deployment.")
async def _grpc_serve_deploy(
    ctx: typer.Context,
    model_name: str = typer.Option(
        ..., "-m", "--model-name", help="Name of the model to use (e.g. stabilityai/stable-diffusion-v2)."
    ),
    min_replicas: int = typer.Option(0, "-min", "--min-replicas", help="Minimum number of replicas to deploy."),
    max_replicas: int = typer.Option(2, "-max", "--max-replicas", help="Maximum number of replicas to deploy."),
    daemon: bool = typer.Option(False, "-d", "--daemon", help="Run the deployment in the background."),
):
    """Create a serve deployment from the model name.

    Usage:
        $ nos grpc-serve deploy -m stabilityai/stable-diffusion-v2
    """
    # Initialize models
    async with grpc.aio.insecure_channel(ctx.obj.address) as channel:
        stub = nos_service_pb2_grpc.InferenceServiceStub(channel)
        # Create init model request
        request = nos_service_pb2.InitModelRequest(model_name=model_name, min_replicas=1)
        # Issue init model request
        response: nos_service_pb2.InitModelResponse = await stub.InitModel(request)
        console.print(response.result)


@serve_grpc_cli.async_command("img2vec", help="Encode image into an embedding.")
async def _predict_img2vec(
    ctx: typer.Context,
    model_name: str = typer.Option(
        "openai/clip-vit-base-patch32",
        "-m",
        "--model-name",
        help="Name of the model to use (e.g. openai/clip-vit-base-patch32).",
    ),
    filename: str = typer.Option(..., "-i", "--input", help="Input image filename."),
    benchmark: int = typer.Option(1, "-b", "--benchmark", help="Run benchmark."),
) -> None:
    from PIL import Image

    img = Image.open(filename)

    st = time.perf_counter()
    with rich.status.Status("[bold green] Generating embedding ...[/bold green]") as status:
        async with grpc.aio.insecure_channel(ctx.obj.address) as channel:
            stub = nos_service_pb2_grpc.InferenceServiceStub(channel)
            try:
                if benchmark > 1:
                    status.stop()
                for _ in tqdm(range(benchmark), disable=benchmark <= 1):
                    response = await stub.Predict(
                        nos_service_pb2.InferenceRequest(
                            method="img2vec",
                            model_name=model_name,
                            image_request=nos_service_pb2.ImageRequest(image_bytes=ray.cloudpickle.dumps(img)),
                        )
                    )
                response = ray.cloudpickle.loads(response.result)
            except grpc.RpcError as e:
                console.print(f"[red] ✗ Failed to encode image (text={e}).[/red]")
                return
    console.print(
        f"[bold green] ✓ Generated embedding ({response['embedding'][..., :4]}..., time={(time.perf_counter() - st) * 1e3/benchmark:.1f}ms) [/bold green]"
    )


@serve_grpc_cli.async_command("txt2vec", help="Generate an embedding from a text prompt.")
async def _predict_txt2vec(
    ctx: typer.Context,
    model_name: str = typer.Option(
        "openai/clip-vit-base-patch32",
        "-m",
        "--model-name",
        help="Name of the model to use (e.g. openai/clip-vit-base-patch32).",
    ),
    prompt: str = typer.Option(
        ..., "-i", "--input", help="Prompt to generate image. (e.g. a cat dancing on the grass.)"
    ),
    benchmark: int = typer.Option(1, "-b", "--benchmark", help="Benchmark the inference time (upto N requests)."),
) -> None:
    st = time.perf_counter()
    with rich.status.Status("[bold green] Generating embedding ...[/bold green]") as status:
        async with grpc.aio.insecure_channel(ctx.obj.address) as channel:
            stub = nos_service_pb2_grpc.InferenceServiceStub(channel)
            try:
                if benchmark > 1:
                    status.stop()
                for _ in tqdm(range(benchmark), disable=benchmark <= 1):
                    response = await stub.Predict(
                        nos_service_pb2.InferenceRequest(
                            method="txt2vec",
                            model_name=model_name,
                            text_request=nos_service_pb2.TextRequest(text=prompt),
                        )
                    )
                response = ray.cloudpickle.loads(response.result)
            except grpc.RpcError as e:
                console.print(f"[red] ✗ Failed to generate image (text={e}).[/red]")
                return
    console.print(
        f"[bold green] ✓ Generated embedding ({response['embedding'][..., :4]}..., time={(time.perf_counter() - st) * 1e3/benchmark:.1f}ms) [/bold green]"
    )


@serve_grpc_cli.async_command("txt2img", help="Generate an image from a text prompt.")
async def _predict_txt2img(
    ctx: typer.Context,
    model_name: str = typer.Option(
        "stabilityai/stable-diffusion-2",
        "-m",
        "--model-name",
        help="Name of the model to use (e.g. stabilityai/stable-diffusion-2).",
    ),
    prompt: str = typer.Option(
        ..., "-i", "--input", help="Prompt to generate image. (e.g. a cat dancing on the grass.)"
    ),
    img_size: int = typer.Option(512, "-s", "--img-size", help="Image size to generate."),
    benchmark: int = typer.Option(1, "-b", "--benchmark", help="Benchmark the inference time (upto N requests)."),
) -> None:
    st = time.perf_counter()
    with rich.status.Status("[bold green] Generating image ...[/bold green]") as status:
        async with grpc.aio.insecure_channel(ctx.obj.address) as channel:
            stub = nos_service_pb2_grpc.InferenceServiceStub(channel)
            try:
                if benchmark > 1:
                    status.stop()
                for _ in tqdm(range(benchmark), disable=benchmark <= 1):
                    response = await stub.Predict(
                        nos_service_pb2.InferenceRequest(
                            method="txt2img",
                            model_name=model_name,
                            text_request=nos_service_pb2.TextRequest(text=prompt),
                        )
                    )
                response = ray.cloudpickle.loads(response.result)
            except grpc.RpcError as e:
                console.print(f"[red] ✗ Failed to generate image (text={e}).[/red]")
                return
    console.print(
        f"[bold green] ✓ Generated image ({response['img']}..., time={(time.perf_counter() - st) * 1e3/benchmark:.1f}ms) [/bold green]"
    )


@serve_grpc_cli.async_command("img2bbox", help="Predict bounding boxes from image.")
async def _predict_img2bbox(
    ctx: typer.Context,
    model_name: str = typer.Option(
        "open-mmlab/faster-rcnn",
        "-m",
        "--model-name",
        help="Name of the model to use (e.g. open-mmlab/efficientdet-d3).",
    ),
    filename: str = typer.Option(..., "-i", "--input", help="Input image filename."),
    benchmark: int = typer.Option(1, "-b", "--benchmark", help="Benchmark the inference time (upto N requests)."),
) -> None:
    from PIL import Image

    img = Image.open(filename)

    with rich.status.Status("[bold green] Predict bounding boxes ...[/bold green]"):
        async with grpc.aio.insecure_channel(ctx.obj.address) as channel:
            stub = nos_service_pb2_grpc.InferenceServiceStub(channel)
            st = time.perf_counter()
            try:
                for _ in tqdm(range(benchmark), disable=True):
                    response = await stub.Predict(
                        nos_service_pb2.InferenceRequest(
                            method="img2bbox",
                            model_name=model_name,
                            image_request=nos_service_pb2.ImageRequest(image_bytes=ray.cloudpickle.dumps(img)),
                        )
                    )
                response = ray.cloudpickle.loads(response.result)
                scores, labels, bboxes = response["bboxes"], response["scores"], response["labels"]
                console.print(
                    f"[bold green] ✓ Predicted bounding boxes (bboxes={bboxes.shape}, scores={scores.shape}, labels={labels.shape}, time={(time.perf_counter() - st) * 1e3/benchmark:.1f}ms) [/bold green]"
                )
            except grpc.RpcError as e:
                console.print(f"[red] ✗ Failed to predict bounding boxes (text={e}).[/red]")
                return
