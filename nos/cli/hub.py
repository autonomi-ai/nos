from pathlib import Path

import humanize
import rich.console
import rich.status
import rich.table
import typer
from rich import print

from nos.common.spec import ModelSpec
from nos.constants import NOS_MODELS_DIR


hub_cli = typer.Typer(name="hub", help="NOS Hub CLI.", no_args_is_help=True)
DEFAULT_MODEL_CACHE_DIR = NOS_MODELS_DIR


@hub_cli.command("list")
def _list_models(private: bool = typer.Option(False, "-p", "--private", help="List private models.")) -> None:
    from nos import hub

    table = rich.table.Table(title="[green]  Models [/green]")
    table.add_column("Model Name")
    table.add_column("Method")
    table.add_column("Task")
    table.add_column("Resources (CPU Memory)")
    table.add_column("Resources (GPU Memory)")

    for model in hub.list(private=private):
        spec: ModelSpec = hub.load_spec(model)
        for method in spec.signature:
            table.add_row(
                f"[green]{model}[/green]",
                method,
                spec.task(method),
                f"{humanize.naturalsize(spec.metadata().resources.memory, binary=True)}",
                f"{humanize.naturalsize(spec.metadata().resources.device_memory, binary=True)}",
            )
    print(table)


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
    from nos import hub

    output_directory = Path(output_directory) / model_name
    output_directory.mkdir(parents=True, exist_ok=True)

    # Downloads the model from the hub and loads it (also caches).
    hub.load(model_name)
    # TODO (spillai): Save model to output_directory
    # model.save_pretrained(output_directory)
