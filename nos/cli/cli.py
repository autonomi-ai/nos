import typer

from nos.cli.hub import hub_cli
from nos.cli.predict import predict_cli
from nos.cli.system import system_cli


app_cli = typer.Typer(no_args_is_help=True)
app_cli.add_typer(hub_cli)
app_cli.add_typer(system_cli)
app_cli.add_typer(predict_cli)

if __name__ == "__main__":
    app_cli()
