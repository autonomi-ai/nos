import typer

from nos.cli.benchmark import benchmark_cli
from nos.cli.docker import docker_cli
from nos.cli.hub import hub_cli
from nos.cli.serve_grpc import serve_grpc_cli
from nos.cli.system import system_cli


app_cli = typer.Typer(no_args_is_help=True)
app_cli.add_typer(hub_cli)
app_cli.add_typer(system_cli)
app_cli.add_typer(benchmark_cli)
app_cli.add_typer(docker_cli)
app_cli.add_typer(serve_grpc_cli)


if __name__ == "__main__":
    app_cli()
