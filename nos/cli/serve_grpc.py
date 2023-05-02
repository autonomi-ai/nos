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

from nos.experimental.grpc import import_module


nos_service_pb2 = import_module("nos_service_pb2")
nos_service_pb2_grpc = import_module("nos_service_pb2_grpc")

serve_grpc_cli = typer.Typer(name="serve-grpc", help="NOS gRPC Serve CLI.", no_args_is_help=True)
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


@serve_grpc_cli.command("list", help="List all gRPC deployments.")
def _grpc_serve_list(ctx: typer.Context):
    """List all gRPC deployments."""

    with grpc.insecure_channel(ctx.obj.address) as channel:
        stub = nos_service_pb2_grpc.InferenceServiceStub(channel)
        try:
            response: nos_service_pb2.ModelListResponse = stub.ListModels(empty_pb2.Empty())
            console.print(response.models)
        except grpc.RpcError as e:
            console.print(f"[red] ✗ Failed to list models ({e}).[/red]")


@serve_grpc_cli.command("img2vec", help="Encode image into an embedding.")
def _predict_img2vec(
    ctx: typer.Context,
    model_name: str = typer.Option(
        "openai/clip-vit-base-patch32",
        "-m",
        "--model-name",
        help="Name of the model to use (e.g. openai/clip-vit-base-patch32).",
    ),
    filename: str = typer.Option(..., "-i", "--input", help="Input image filename."),
) -> None:
    from PIL import Image

    img = Image.open(filename)

    st = time.perf_counter()
    with rich.status.Status("[bold green] Generating embedding ...[/bold green]"):
        with grpc.insecure_channel(ctx.obj.address) as channel:
            stub = nos_service_pb2_grpc.InferenceServiceStub(channel)
            try:
                response = stub.Predict(
                    nos_service_pb2.InferenceRequest(
                        method="img2vec",
                        model_name=model_name,
                        image_request=nos_service_pb2.ImageRequest(image_bytes=ray.cloudpickle.dumps(img)),
                    )
                )
                response = ray.cloudpickle.loads(response.result)
            except grpc.RpcError as exc:
                console.print(f"[red] ✗ Failed to encode image. [/red]\n[bold white]{exc}[/bold white]")
                return
    console.print(
        f"[bold green] ✓ Generated embedding ((1, {response['embedding'].shape[-1]}), time=~{(time.perf_counter() - st) * 1e3:.1f}ms) [/bold green]"
    )


@serve_grpc_cli.command("txt2vec", help="Generate an embedding from a text prompt.")
def _predict_txt2vec(
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
) -> None:
    st = time.perf_counter()
    with rich.status.Status("[bold green] Generating embedding ...[/bold green]"):
        with grpc.insecure_channel(ctx.obj.address) as channel:
            stub = nos_service_pb2_grpc.InferenceServiceStub(channel)
            try:
                response = stub.Predict(
                    nos_service_pb2.InferenceRequest(
                        method="txt2vec",
                        model_name=model_name,
                        text_request=nos_service_pb2.TextRequest(text=prompt),
                    )
                )
                response = ray.cloudpickle.loads(response.result)
            except grpc.RpcError as exc:
                console.print(f"[red] ✗ Failed to generate image. [/red]\n[bold white]{exc}[/bold white]")
                return
    console.print(
        f"[bold green] ✓ Generated embedding ({response['embedding'][..., :].shape}..., time=~{(time.perf_counter() - st) * 1e3:.1f}ms) [/bold green]"
    )


@serve_grpc_cli.command("txt2img", help="Generate an image from a text prompt.")
def _predict_txt2img(
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
) -> None:
    st = time.perf_counter()
    with rich.status.Status("[bold green] Generating image ...[/bold green]"):
        with grpc.insecure_channel(ctx.obj.address) as channel:
            stub = nos_service_pb2_grpc.InferenceServiceStub(channel)
            try:
                response = stub.Predict(
                    nos_service_pb2.InferenceRequest(
                        method="txt2img",
                        model_name=model_name,
                        text_request=nos_service_pb2.TextRequest(text=prompt),
                    )
                )
                response = ray.cloudpickle.loads(response.result)
            except grpc.RpcError as exc:
                console.print(f"[red] ✗ Failed to generate image. [/red]\n[bold white]{exc}[/bold white]")
                return
    console.print(
        f"[bold green] ✓ Generated image ({response['image']}..., time=~{(time.perf_counter() - st) * 1e3:.1f}ms) [/bold green]"
    )


@serve_grpc_cli.command("img2bbox", help="Predict bounding boxes from image.")
def _predict_img2bbox(
    ctx: typer.Context,
    model_name: str = typer.Option(
        "torchvision/fasterrcnn_mobilenet_v3_large_320_fpn",
        "-m",
        "--model-name",
        help="Name of the model to use (e.g. torchvision/fasterrcnn_mobilenet_v3_large_320_fpn).",
    ),
    filename: str = typer.Option(..., "-i", "--input", help="Input image filename."),
) -> None:
    from PIL import Image

    img = Image.open(filename).resize((640, 480))
    with rich.status.Status("[bold green] Predict bounding boxes ...[/bold green]"):
        with grpc.insecure_channel(ctx.obj.address) as channel:
            stub = nos_service_pb2_grpc.InferenceServiceStub(channel)
            st = time.perf_counter()
            try:
                response = stub.Predict(
                    nos_service_pb2.InferenceRequest(
                        method="img2bbox",
                        model_name=model_name,
                        image_request=nos_service_pb2.ImageRequest(image_bytes=ray.cloudpickle.dumps(img)),
                    )
                )
                response = ray.cloudpickle.loads(response.result)
                scores, labels, bboxes = response["bboxes"], response["scores"], response["labels"]
                console.print(
                    f"[bold green] ✓ Predicted bounding boxes (bboxes={bboxes.shape}, scores={scores.shape}, labels={labels.shape}, time=~{(time.perf_counter() - st) * 1e3:.1f}ms) [/bold green]"
                )
            except grpc.RpcError as exc:
                console.print(f"[red] ✗ Failed to predict bounding boxes. [/red]\n[bold white]{exc}[/bold white]")
                return
