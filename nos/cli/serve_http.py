"""REST-based Client CLI for NOS.

Usage:
    $ nos serve-grpc --help
    $ nos serve-grpc list
    $ nos serve-grpc deploy -m stabilityai/stable-diffusion-v2
    $ nos serve-grpc img2vec -i tests/test_data/test.jpg
    $ nos serve-grpc txt2vec -i 'Hello World!'
    $ nos serve-grpc txt2img -i 'a cat dancing on the grass'

"""
import time

import rich.console
import rich.status
import rich.table
import typer

from nos.experimental.http.client import NOS_SERVE_DEFAULT_HTTP_HOST, NOS_SERVE_DEFAULT_HTTP_PORT, SimpleClient
from nos.experimental.http.service import PredictionRequest, PredictionResponse


serve_cli = typer.Typer(name="serve", help="NOS Serve CLI.", no_args_is_help=True)


@serve_cli.command("list", help="List all deployments.")
def _serve_list():
    """List all deployments."""
    from nos import serve

    console = rich.console.Console()
    table = rich.table.Table(title="Deployments")
    table.add_column("Name", justify="left", style="cyan")
    table.add_column("Model", justify="left", style="magenta")
    table.add_column("Status", justify="left", style="green")
    table.add_column("URL", justify="left", style="blue")

    with rich.status.Status("Listing deployments ..."):
        deployments = serve.list()
        if deployments is None:
            console.print("[red]nos serve is not running, did you run `nos serve ...`?.")
            return

        for deployment in serve.list():
            table.add_row(deployment.name, deployment.model_name, deployment.status, deployment.url)


@serve_cli.command("deploy", help="Create a deployment.")
def _serve_deploy(
    model_name: str = typer.Option(
        ..., "-m", "--model-name", help="Name of the model to use (e.g. stabilityai/stable-diffusion-v2)."
    ),
    min_replicas: int = typer.Option(0, "-min", "--min-replicas", help="Minimum number of replicas to deploy."),
    max_replicas: int = typer.Option(2, "-max", "--max-replicas", help="Maximum number of replicas to deploy."),
    host: str = typer.Option(NOS_SERVE_DEFAULT_HTTP_HOST, "-h", "--host", help="Host of the NOS backend."),
    port: int = typer.Option(NOS_SERVE_DEFAULT_HTTP_PORT, "-p", "--port", help="Port of the NOS backend."),
    daemon: bool = typer.Option(False, "-d", "--daemon", help="Run the deployment in the background."),
):
    """Create a serve deployment from the model name.

    Usage:
        $ nos serve deploy -m stabilityai/stable-diffusion-v2
    """
    from nos import hub, serve

    # Get the model handle by name
    model_spec = hub.load_spec(model_name)

    # Create a deployment from the model handle
    serve.deployment(
        model_name,
        model_spec,
        deployment_config={
            "ray_actor_options": {"num_gpus": 1},
            "autoscaling_config": {"min_replicas": min_replicas, "max_replicas": max_replicas},
        },
        host=host,
        port=port,
        daemon=daemon,
    )


def wait_for_backend(host: str, port: int, timeout: int = 30) -> None:
    """Wait for the backend to be ready."""
    client = SimpleClient(host, port)
    with rich.status.Status("[bold green] Waiting for nos backend ...[/bold green]]") as status:
        try:
            client.wait(timeout=timeout)
            status.update(f"[bold green] Connected to http://{host}:{port}.[/bold green]")
        except TimeoutError:
            status.update("[red] nos serve is not running, did you run `nos serve ...`?.")
            return


@serve_cli.command("txt2img", help="Generate an image from a text prompt.")
def _predict_txt2img(
    prompt: str = typer.Option(
        ..., "-i", "--input", help="Prompt to generate image. (e.g. a cat dancing on the grass.)"
    ),
    img_size: int = typer.Option(512, "-s", "--img-size", help="Image size to generate."),
    host: str = typer.Option(NOS_SERVE_DEFAULT_HTTP_HOST, "-h", "--host", help="Host of the NOS backend."),
    port: int = typer.Option(NOS_SERVE_DEFAULT_HTTP_PORT, "-p", "--port", help="Port of the NOS backend."),
) -> None:
    wait_for_backend(host, port)

    console = rich.console.Console()
    client = SimpleClient(host, port)

    st = time.perf_counter()
    with rich.status.Status("[bold green] Generating image ...[/bold green]"):
        request = PredictionRequest(
            method="txt2img", request={"prompt": prompt, "height": img_size, "width": img_size}
        )
        resp = client.post("predict", json=request.dict())
        if resp.status_code != 201:
            console.print(f"[red] ✗ Failed to generate image (text={resp.text}).[/red]")
            return
        with open("output.png", "wb") as f:
            f.write(resp.content)
    console.print(f"[bold green] ✓ Generated image (output.png, time={time.perf_counter() - st:.3f}s) [/bold green]")


@serve_cli.command("txt2vec", help="Generate an embedding from a text prompt.")
def _predict_txt2vec(
    prompt: str = typer.Option(
        ..., "-i", "--input", help="Prompt to generate image. (e.g. a cat dancing on the grass.)"
    ),
    host: str = typer.Option(NOS_SERVE_DEFAULT_HTTP_HOST, "-h", "--host", help="Host of the NOS backend."),
    port: int = typer.Option(NOS_SERVE_DEFAULT_HTTP_PORT, "-p", "--port", help="Port of the NOS backend."),
) -> None:
    wait_for_backend(host, port)

    console = rich.console.Console()
    client = SimpleClient(host, port)
    st = time.perf_counter()
    with rich.status.Status("[bold green] Generating embedding ...[/bold green]"):
        request = PredictionRequest(method="txt2vec", request={"text": prompt})
        response = client.post("predict", json=request.dict())
        if response.status_code != 201:
            console.print(f"[red] ✗ Failed to generate image (text={response.text}).[/red]")
            return
        response = PredictionResponse.parse_obj(response.json())
    console.print(
        f"[bold green] ✓ Generated embedding ({response}, time={time.perf_counter() - st:.3f}s) [/bold green]"
    )
