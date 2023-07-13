import rich.console
import rich.status
import typer

import docker
from nos.server import InferenceServiceRuntime


docker_cli = typer.Typer(name="docker", help="NOS Docker CLI.", no_args_is_help=True)
console = rich.console.Console()


def get_running_inference_service_runtime_container(id: str = None) -> docker.models.containers.Container:
    """Get running container."""
    filters = {"status": "running"}
    if id is not None:
        filters["id"] = id
    containers = InferenceServiceRuntime.list(filters=filters)
    if not len(containers):
        console.print("[bold red] ✗ No inference server running.[/bold red]")
        return None
    if len(containers) > 1 and id is None:
        console.print("[bold red] ✗ Multiple inference servers running, provide an id.[/bold red]")
        return None
    (container,) = containers
    return container


@docker_cli.command("list", help="List running NOS inference servers.")
def _docker_list():
    """List running docker inference servers.

    Usage:
        $ nos docker list
    """
    containers = InferenceServiceRuntime.list()
    if len(containers) == 0:
        console.print("[bold red] ✗ No inference server running.[/bold red]")
        return
    console.print("[bold green] ✓ Inference servers running:[/bold green]")
    for container in containers:
        console.print(
            f"    - id={container.id[:12]} name={container.name: <40} status={container.status}  image={container.image}"
        )


@docker_cli.command("start", help="Start NOS inference server.")
def _docker_start(
    runtime: str = typer.Option("cpu", "--runtime", help="The runtime to use (cpu or gpu)."),
):
    """Start the NOS inference server.

    Usage:
        $ nos docker start
    """
    if runtime not in InferenceServiceRuntime.configs:
        raise ValueError(
            f"Invalid inference runtime: {runtime}, available: {list(InferenceServiceRuntime.configs.keys())}"
        )

    containers = InferenceServiceRuntime.list()
    if len(containers) > 0:
        container = containers[0]
        console.print(
            f"[bold yellow] ✗ Inference server already running (name={container.name}, id={container.id[:12]}).[/bold yellow]"
        )
        return
    with rich.status.Status("[bold green] Starting inference server ...[/bold green]") as status:
        runtime = InferenceServiceRuntime(runtime=runtime)
        if runtime.get_container_status() == "running":
            status.stop()
            id = runtime.get_container_id()
            console.print(f"[bold green] ✓ Inference server already running (id={id[:12]}).[/bold green]")
            return
        runtime.start()
        id = runtime.get_container_id()
    console.print(f"[bold green] ✓ Inference server started (id={id[:12]}). [/bold green]")


@docker_cli.command("stop", help="Stop NOS inference engine.")
def _docker_stop(
    id: str = typer.Option(None, "--id", help="The id of the inference server container."),
):
    """Stop the docker inference servers.

    Usage:
        $ nos docker stop
        $ nos docker stop --id <id>
    """
    container = get_running_inference_service_runtime_container(id=id)
    if container is None:
        return
    id = container.id
    with rich.status.Status("[bold green] Stopping inference server ...[/bold green]"):
        try:
            container.remove(force=True)
        except Exception:
            console.print("[bold red] ✗ Failed to stop inference server.[/bold red]")
            return
    console.print(f"[bold green] ✓ Inference server stopped (id={id[:12]}).[/bold green]")


@docker_cli.command("logs", help="Get NOS inference server logs.")
def _docker_logs(
    id: str = typer.Option(None, "--id", help="The id of the inference server container."),
):
    """Get the docker logs of the inference server.

    Usage:
        $ nos docker logs <id>
    """
    container = get_running_inference_service_runtime_container(id=id)
    if container is None:
        return
    id = container.id
    console.print(f"[bold green] Fetching server logs (id={id[:12]}) ...[/bold green]")
    for line in container.logs(stream=True):
        print(line.decode("utf-8").strip())
