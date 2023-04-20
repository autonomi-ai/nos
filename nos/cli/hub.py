from pathlib import Path

import typer

from nos.constants import NOS_MODELS_DIR


DEFAULT_MODEL_CACHE_DIR = NOS_MODELS_DIR


hub_cli = typer.Typer(name="hub", help="NOS Hub CLI.", no_args_is_help=True)


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
    import torch

    if model_name == "stabilityai/stable-diffusion-2-fp16":
        from diffusers import StableDiffusionPipeline

        output_directory = Path(output_directory) / model_name
        output_directory.mkdir(parents=True, exist_ok=True)

        print("Downloading stable-diffusion-2-fp16 model...")
        StableDiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1-base",
            revision="fp16",
            torch_dtype=torch.float16,
            use_auth_token=token if token != "" else True,
        ).save_pretrained(output_directory)

    else:
        raise NotImplementedError(f"Model {model_name} not implemented.")
