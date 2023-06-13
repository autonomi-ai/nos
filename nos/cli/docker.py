import rich.console
import rich.status
import typer

from nos.server.runtime import InferenceServiceRuntime


docker_cli = typer.Typer(name="docker", help="NOS Docker CLI.", no_args_is_help=True)
console = rich.console.Console()


@docker_cli.command("start", help="Start NOS inference engine.")
def _docker_start(
    gpu: bool = typer.Option(False, "--gpu", help="Start the container with GPU support."),
):
    """Start the NOS inference engine.

    Usage:
        $ nos docker start
    """
    with rich.status.Status("[bold green] Starting inference client ...[/bold green]") as status:
        runtime = "gpu" if gpu else "cpu"
        runtime = InferenceServiceRuntime(runtime=runtime, name=f"nos-inference-service-runtime-{runtime}")
        if runtime.get_container_status() == "running":
            status.stop()
            id = runtime.get_container_id()
            console.print(
                f"[bold green] ✓ Inference client already running (id={id[:12] if id else None}).[/bold green]"
            )
            return
        runtime.start()
        id = runtime.get_container_id()
    console.print(f"[bold green] ✓ Inference client started (id={id[:12] if id else None}). [/bold green]")


@docker_cli.command("stop", help="Stop NOS inference engine.")
def _docker_stop():
    """Stop the docker inference engine.

    Usage:
        $ nos docker stop
    """
    with rich.status.Status("[bold green] Stopping inference client ...[/bold green]"):
        client = InferenceServiceRuntime()
        client.stop()
    console.print("[bold green] ✓ Inference client stopped.[/bold green]")


@docker_cli.command("logs", help="Get NOS inference engine logs.")
def _docker_logs():
    """Get the docker logs of the inference engine.

    Usage:
        $ nos docker logs
    """
    client = InferenceServiceRuntime()
    id = client.id()
    with rich.status.Status(f"[bold green] Fetching client logs (id={id[:12] if id else None}) ...[/bold green]"):
        logs = client.get_logs()
    print(logs)
