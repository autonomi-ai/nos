import time
from dataclasses import dataclass

import grpc
import rich.console
import rich.status
import rich.table
import typer
from google.protobuf import empty_pb2

from nos.experimental.grpc import import_module
from nos.serve.client import NOS_SERVE_DEFAULT_HTTP_HOST, NOS_SERVE_DEFAULT_HTTP_PORT, SimpleClient


nos_service_pb2 = import_module("nos_service_pb2")
nos_service_pb2_grpc = import_module("nos_service_pb2_grpc")


grpc_serve_cli = typer.Typer(name="grpc-serve", help="NOS gRPC Serve CLI.", no_args_is_help=True)


@dataclass
class gRPCConfig:
    """Common gRPC options"""

    address: str


@grpc_serve_cli.callback()
def grpc_config(
    ctx: typer.Context,
    address: str = typer.Option("localhost:50051", "-a", "--address", help="Address of the gRPC server."),
):
    """Common gRPC options"""
    ctx.obj = gRPCConfig(address)

    # TOOD (spillai): Ping the gRPC server otherwise raise an error


def wait_for_grpc_backend(address: str, timeout: int = 10):
    """Wait for the gRPC backend to be ready."""
    console = rich.console.Console()
    with grpc.insecure_channel(address) as channel:
        stub = nos_service_pb2_grpc.InferenceServiceStub(channel)

        # Wait for backend to be ready
        start = time.time()
        while time.time() - start < timeout:
            try:
                request = empty_pb2.Empty()
                stub.ListModels(request)
                return
            except grpc.RpcError as e:
                console.print(f"[red] ✗ Failed to list models ({e}).[/red]")
                time.sleep(1)

        raise RuntimeError(f"Failed to connect to gRPC server at {address}.")


@grpc_serve_cli.command("list", help="List all gRPC deployments.")
def _grpc_serve_list(ctx: typer.Context):
    """List all gRPC deployments."""

    console = rich.console.Console()
    with grpc.insecure_channel(ctx.obj.address) as channel:
        stub = nos_service_pb2_grpc.InferenceServiceStub(channel)

        # List models
        try:
            request = empty_pb2.Empty()
            response: nos_service_pb2.ModelListResponse = stub.ListModels(request)
            print(response.models)
        except grpc.RpcError as e:
            console.print(f"[red] ✗ Failed to list models ({e}).[/red]")


@grpc_serve_cli.command("deploy", help="Create a deployment.")
def _grpc_serve_deploy(
    ctx: typer.Context,
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
        $ nos grpc-serve deploy -m stabilityai/stable-diffusion-v2
    """

    # TODO (spillai): Deploy the gRPC server here in the background (as a docker daemon)
    # Wait for backend to be ready
    # wait_for_backend(ctx.obj.address)

    # Initialize models
    with grpc.insecure_channel(ctx.obj.address) as channel:
        stub = nos_service_pb2_grpc.InferenceServiceStub(channel)

        # Create init model request
        request = nos_service_pb2.InitModelRequest(model_name=model_name, min_replicas=0, max_replicas=2)

        # Issue init model request
        response: nos_service_pb2.InitModelResponse = stub.InitModel(request)
        print(response.result)


@grpc_serve_cli.command("txt2vec", help="Generate an embedding from a text prompt.")
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
    host: str = typer.Option(NOS_SERVE_DEFAULT_HTTP_HOST, "-h", "--host", help="Host of the NOS backend."),
    port: int = typer.Option(NOS_SERVE_DEFAULT_HTTP_PORT, "-p", "--port", help="Port of the NOS backend."),
) -> None:
    # wait_for_backend(host, port)

    console = rich.console.Console()
    SimpleClient(host, port)
    st = time.perf_counter()
    with rich.status.Status("[bold green] Generating embedding ...[/bold green]"):
        with grpc.insecure_channel(ctx.obj.address) as channel:
            stub = nos_service_pb2_grpc.InferenceServiceStub(channel)

            # Run inference
            try:
                response = stub.Predict(
                    nos_service_pb2.InferenceRequest(
                        method="txt2vec",
                        model_name=model_name,
                        text=prompt,
                    )
                )
            except grpc.RpcError as e:
                console.print(f"[red] ✗ Failed to generate image (text={e}).[/red]")
                return
            print(response)
    console.print(
        f"[bold green] ✓ Generated embedding ({response}, time={time.perf_counter() - st:.3f}s) [/bold green]"
    )
