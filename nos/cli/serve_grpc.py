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

import rich.console
import rich.status
import rich.table
import typer

from nos.client import InferenceClient
from nos.client.exceptions import NosClientException


serve_grpc_cli = typer.Typer(name="serve-grpc", help="NOS gRPC Serve CLI.", no_args_is_help=True)
console = rich.console.Console()


@dataclass
class gRPCConfig:
    """Common gRPC options"""

    address: str
    client: InferenceClient


@serve_grpc_cli.callback()
def grpc_config(
    ctx: typer.Context,
    address: str = typer.Option("localhost:50051", "-a", "--address", help="Address of the gRPC server."),
):
    """Common gRPC options"""
    client = InferenceClient(address)
    ctx.obj = gRPCConfig(address, client)
    # TODO (spillai): Deploy the gRPC server here in the background (as a docker daemon)
    # TOOD (spillai): Ping the gRPC server otherwise raise an error


@serve_grpc_cli.command("list", help="List all gRPC deployments.")
def _grpc_serve_list(ctx: typer.Context):
    """List all gRPC deployments."""
    try:
        models = ctx.obj.client.ListModels()
        console.print(models)
    except NosClientException as exc:
        console.print(f"[red] ✗ Failed to list models ({exc}).[/red]")


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
        try:
            response = ctx.obj.client.Predict(method="img2vec", model_name=model_name, img=img)
        except NosClientException as exc:
            console.print(f"[red] ✗ Failed to encode image. [/red]\n[bold red]{exc}[/bold red]")
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
        try:
            response = ctx.obj.client.Predict(method="txt2vec", model_name=model_name, text=prompt)
        except NosClientException as exc:
            console.print(f"[red] ✗ Failed to generate image. [/red]\n[bold red]{exc}[/bold red]")
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
        try:
            response = ctx.obj.client.Predict(method="txt2img", model_name=model_name, text=prompt)
        except NosClientException as exc:
            console.print(f"[red] ✗ Failed to generate image. [/red]\n[bold red]{exc}[/bold red]")
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

        st = time.perf_counter()
        try:
            response = ctx.obj.client.Predict(
                method="img2bbox",
                model_name=model_name,
                img=img,
            )
            scores, labels, bboxes = response["bboxes"], response["scores"], response["labels"]
            console.print(
                f"[bold green] ✓ Predicted bounding boxes (bboxes={bboxes.shape}, scores={scores.shape}, labels={labels.shape}, time=~{(time.perf_counter() - st) * 1e3:.1f}ms) [/bold green]"
            )
        except NosClientException as exc:
            console.print(f"[red] ✗ Failed to predict bounding boxes. [/red]\n[bold red]{exc}[/bold red]")
            return
