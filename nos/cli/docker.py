import typer

from nos.experimental.grpc.client import NOS_GRPC_SERVER_CMD, InferenceRuntime
from nos.logging import logger


docker_cli = typer.Typer(name="docker", help="NOS Docker CLI.", no_args_is_help=True)


@docker_cli.command("start", help="Start NOS inference engine.")
def _docker_start(
    command: str = typer.Option(NOS_GRPC_SERVER_CMD, "-c", "--command", help="Command to run in the container."),
    gpu: bool = typer.Option(False, "--gpu", help="Start the container with GPU support."),
):
    """Start the NOS inference engine.

    Usage:
        $ nos docker start
    """
    runtime = InferenceRuntime()
    runtime.start(detach=True, gpu=gpu)
    logger.info("NOS inference engine started.")


@docker_cli.command("stop", help="Stop NOS inference engine.")
def _docker_stop():
    """Stop the docker inference engine.

    Usage:
        $ nos docker stop
    """
    runtime = InferenceRuntime()
    runtime.stop()


@docker_cli.command("logs", help="Get NOS inference engine logs.")
def _docker_logs():
    """Get the docker logs of the inference engine.

    Usage:
        $ nos docker logs
    """

    runtime = InferenceRuntime()
    logs = runtime.get_logs()
    print(logs)
