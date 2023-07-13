import typer
from rich.console import Console
from rich.json import JSON
from rich.panel import Panel

from docker.errors import APIError
from nos.common.system import get_system_info, has_gpu
from nos.logging import logger
from nos.server import DockerRuntime


system_cli = typer.Typer(name="system", help="NOS System CLI.", no_args_is_help=True)
console = Console()


@system_cli.command("info")
def _system_info() -> None:
    """Get system information (including CPU, GPU, RAM, etc.)"""

    # Get system info
    console.print(Panel(JSON.from_data(get_system_info(docker=True, gpu=True)), title="System"))

    if not has_gpu():
        console.print("No GPU detected, exiting early.")
        return
    else:
        console.print("GPU detected, fetching nvidia-smi information within docker.")

    # Check if GPU is available
    CUDA_RUNTIME_IMAGE = "nvidia/cuda:11.8.0-base-ubuntu22.04"
    nvidia_docker_gpu_info = ""

    runtime = DockerRuntime()
    try:
        # Get the output of nvidia-smi running in the container
        container = runtime.start(
            image=CUDA_RUNTIME_IMAGE, name="nos-server-gpu-test", command="nvidia-smi", detach=True, gpu=True
        )
        for i, log in enumerate(container.logs(stream=True)):
            nvidia_docker_gpu_info += "\n" if i > 0 else ""
            nvidia_docker_gpu_info += f"{log.strip().decode()}"
        container.stop()
    except (APIError, ModuleNotFoundError, Exception) as exc:
        logger.error(f"Failed to run nvidia-smi within docker container: {exc}")
        nvidia_docker_gpu_info = "Failed to run nvidia-smi within docker container"
    console.print(Panel(nvidia_docker_gpu_info, title="nvidia-smi (docker)"))
