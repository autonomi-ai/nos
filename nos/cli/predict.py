"""gRPC-based Client CLI for NOS.

Usage:
    $ nos predict --help
    $ nos predict list
    $ nos predict img2vec -i tests/test_data/test.jpg
    $ nos predict txt2vec -i 'Hello World!'
    $ nos predict txt2img -i 'a cat dancing on the grass'

"""
import time
from dataclasses import dataclass

import rich.console
import rich.status
import rich.table
import typer

from nos.client import InferenceClient
from nos.common import TaskType
from nos.common.exceptions import NosClientException


predict_cli = typer.Typer(name="predict", help="NOS gRPC Serve CLI.", no_args_is_help=True)
console = rich.console.Console()


@dataclass
class gRPCConfig:
    """Common gRPC options"""

    address: str
    client: InferenceClient


@predict_cli.callback()
def grpc_config(
    ctx: typer.Context,
    address: str = typer.Option("[::]:50051", "-a", "--address", help="Address of the gRPC server."),
):
    """Common gRPC options"""
    client = InferenceClient(address)
    ctx.obj = gRPCConfig(address, client)
    # TODO (spillai): Deploy the gRPC server here in the background (as a docker daemon)
    # TOOD (spillai): Ping the gRPC server otherwise raise an error


@predict_cli.command("list", help="List all gRPC deployments.")
def _list_models(ctx: typer.Context):
    """List all gRPC deployments."""
    try:
        models = ctx.obj.client.ListModels()
        console.print(models)
    except NosClientException as exc:
        console.print(f"[red] ✗ Failed to list models ({exc}).[/red]")


@predict_cli.command("img2vec", help="Encode image into an embedding.")
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

    with rich.status.Status("[bold green] Generating embedding ...[/bold green]"):
        try:
            st = time.perf_counter()
            response = ctx.obj.client.Run(task=TaskType.IMAGE_EMBEDDING, model_name=model_name, images=[img])
            end = time.perf_counter()
        except NosClientException as exc:
            console.print(f"[red] ✗ Failed to encode image. [/red]\n[bold red]{exc}[/bold red]")
            return
    console.print(
        f"[bold green] ✓ Generated embedding ((1, {response['embedding'].shape[-1]}), time=~{(end - st) * 1e3:.1f}ms) [/bold green]"
    )


@predict_cli.command("txt2vec", help="Generate an embedding from a text prompt.")
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
    with rich.status.Status("[bold green] Generating embedding ...[/bold green]"):
        try:
            st = time.perf_counter()
            response = ctx.obj.client.Run(task=TaskType.TEXT_EMBEDDING, model_name=model_name, texts=[prompt])
            end = time.perf_counter()
        except NosClientException as exc:
            console.print(f"[red] ✗ Failed to generate image. [/red]\n[bold red]{exc}[/bold red]")
            return
    console.print(
        f"[bold green] ✓ Generated embedding ({response['embedding'][..., :].shape}..., time=~{(end - st) * 1e3:.1f}ms) [/bold green]"
    )


@predict_cli.command("txt2img", help="Generate an image from a text prompt.")
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
    num_images: int = typer.Option(1, "-n", "--num-images", help="Number of images to generate."),
) -> None:
    with rich.status.Status("[bold green] Generating image ...[/bold green]"):
        try:
            st = time.perf_counter()
            response = ctx.obj.client.Run(
                task=TaskType.IMAGE_GENERATION,
                model_name=model_name,
                prompts=[prompt],
                height=img_size,
                width=img_size,
                num_images=num_images,
            )
            end = time.perf_counter()
        except NosClientException as exc:
            console.print(f"[red] ✗ Failed to generate image. [/red]\n[bold red]{exc}[/bold red]")
            return
    console.print(
        f"[bold green] ✓ Generated image ({response['images']}..., time=~{(end - st) * 1e3:.1f}ms) [/bold green]"
    )


@predict_cli.command("img2bbox", help="Predict bounding boxes from image.")
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
        try:
            st = time.perf_counter()
            response = ctx.obj.client.Run(task=TaskType.OBJECT_DETECTION_2D, model_name=model_name, images=[img])
            scores, labels, bboxes = response["bboxes"], response["scores"], response["labels"]
            end = time.perf_counter()
            console.print(
                f"[bold green] ✓ Predicted bounding boxes (bboxes={bboxes[0].shape}, scores={scores[0].shape}, labels={labels[0].shape}, time=~{(end - st) * 1e3:.1f}ms) [/bold green]"
            )
        except NosClientException as exc:
            console.print(f"[red] ✗ Failed to predict bounding boxes. [/red]\n[bold red]{exc}[/bold red]")
            return


@predict_cli.command("segmentation", help="Propose a zero-shot segmentation for this image.")
def _predict_segmentation(
    ctx: typer.Context,
    model_name: str = typer.Option(
        "facebook/sam-vit-large",
        "-m",
        "--model-name",
        help="Name of the model to use (e.g. torchvision/fasterrcnn_mobilenet_v3_large_320_fpn).",
    ),
    filename: str = typer.Option(..., "-i", "--input", help="Input image filename."),
) -> None:
    from PIL import Image

    img = Image.open(filename).resize((640, 480))
    with rich.status.Status("[bold green] Predict segmentations ...[/bold green]"):
        try:
            st = time.perf_counter()
            response = ctx.obj.client.Run(task=TaskType.IMAGE_SEGMENTATION_2D, model_name=model_name, images=[img])
            time.perf_counter()
        except NosClientException as exc:
            console.print(f"[red] ✗ Failed to predict segmentations. [/red]\n[bold red]{exc}[/bold red]")
            return

    console.print(
        f"[bold green] ✓ Generated masks ({response['masks']}..., time=~{(time.perf_counter() - st) * 1e3:.1f}ms) [/bold green]"
    )
