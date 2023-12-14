import typer

from nos.cli.hub import hub_cli
from nos.cli.predict import predict_cli
from nos.cli.serve import serve_cli
from nos.cli.system import system_cli
from nos.logging import logger


app_cli = typer.Typer(
    name="nos",
    help="🔥 NOS CLI - Nitrous Oxide for your AI infrastructure.",
    no_args_is_help=True,
)

app_cli.add_typer(hub_cli)
app_cli.add_typer(system_cli)
app_cli.add_typer(predict_cli)
app_cli.add_typer(serve_cli)

# TOFIX (spillai): Currently, the profiler requires the `torch` package
# which shouldn't be a requirement for the CLI. We should move the profiler
# to a separate package and import it here if it's available.
try:
    from nos.cli.profile import profile_cli

    app_cli.add_typer(profile_cli)
except ImportError:

    logger.warning("Could not import profile_cli")


if __name__ == "__main__":
    app_cli()
