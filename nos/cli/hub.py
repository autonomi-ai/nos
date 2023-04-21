from pathlib import Path

import rich.console
import rich.status
import rich.table
import typer

from nos import hub
from nos.constants import NOS_MODELS_DIR


DEFAULT_MODEL_CACHE_DIR = NOS_MODELS_DIR


hub_cli = typer.Typer(name="hub", help="NOS Hub CLI.", no_args_is_help=True)


@hub_cli.command("list")
def _list_models(private: bool = typer.Option(False, "-p", "--private", help="List private models.")) -> None:
    console = rich.console.Console()
    with rich.status.Status("Fetching models from registry ..."):
        console.print(hub.list(private=private))


@hub_cli.command("download")
def _download_hf_model(
    model_name: str = typer.Option(..., "-m", "--model-name", help="Huggingface model name."),
    token: str = typer.Option("", "-t", "--token", help="HF access token"),
    output_directory: str = typer.Option(
        DEFAULT_MODEL_CACHE_DIR,
        "-d",
        "--output-directory",
        help="Download huggingface models locally.",
    ),
) -> None:

    output_directory = Path(output_directory) / model_name
    output_directory.mkdir(parents=True, exist_ok=True)

    # Downloads the model from the hub and loads it (also caches).
    hub.load(model_name)
    # TODO (spillai): Save model to output_directory
    # model.save_pretrained(output_directory)
