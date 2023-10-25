"""
NOS Serve CLI.

Usage:
    cd ~/examples/whisperx/
    nos serve -c config.yaml --local

"""
import subprocess
from dataclasses import asdict, field
from pathlib import Path
from typing import Union

import typer
from pydantic.dataclasses import dataclass
from rich import print
from rich.console import Console
from rich.tree import Tree


serve_cli = typer.Typer(name="serve", help="NOS Serve CLI.", no_args_is_help=True)
console = Console()


@dataclass
class ServeOptions:
    """Render options for docker-compose.yml.j2."""

    config: Union[str, Path]
    """Path to the YAML configuration file."""

    image: str
    """Image name to use for the server."""

    gpu: bool = field(default=False)
    """Whether to use GPU for the server."""

    http: bool = field(default=False)
    """Whether to use HTTP for the server."""

    http_port: int = field(default=8000)
    """HTTP port to use for the server."""

    http_max_workers: int = field(default=4)
    """HTTP max workers to use for the server."""

    logging_level: str = field(default="INFO")
    """Logging level to use for the server."""

    daemon: bool = field(default=False)
    """Whether to run the server in daemon mode."""


@serve_cli.callback(invoke_without_command=True)
def _serve(
    config_filename: str = typer.Option(..., "-c", "--config", help="Serve configuration filename."),
    model: str = typer.Option(None, "-m", "--model", help="Serve a specific model.", show_default=False),
    target: str = typer.Option(None, "--target", help="Serve a specific target.", show_default=False),
    tag: str = typer.Option("{name}:{target}", "--tag", "-t", help="Image tag f-string.", show_default=True),
    # sandbox_path: str = typer.Option(
    #     "/app/serve", "--sandbox-path", help="Sandbox path to use.", show_default=True
    # ),
    http: bool = typer.Option(False, "--http", help="Serve with HTTP gateway.", show_default=True),
    http_port: int = typer.Option(8000, "--http-port", help="HTTP port to use.", show_default=True),
    http_max_workers: int = typer.Option(4, "--http-max-workers", help="HTTP max workers.", show_default=True),
    logging_level: str = typer.Option("INFO", "--logging-level", help="Logging level.", show_default=True),
    daemon: bool = typer.Option(False, "--daemon", help="Run in daemon mode.", show_default=True),
    prod: bool = typer.Option(
        False,
        "-p",
        "--prod",
        help="Run with production flags (slimmer images, no dev. dependencies).",
        show_default=False,
    ),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Verbose output.", show_default=False),
) -> None:
    """Main entrypoint for nos serving (either locally or in the cloud)."""
    from agipack.builder import AGIPack
    from agipack.config import AGIPackConfig
    from jinja2 import Environment, FileSystemLoader

    from nos.common.system import docker_compose_command, has_docker, has_gpu
    from nos.logging import logger, redirect_stdout_to_logger

    # NOTE: `nos serve` does a few things to make it easy to deploy
    # custom models within a newly minted docker runtime environment.
    #
    # The config is broken up into two parts:
    # 1. images:
    #   This section defines the docker images to build, and the
    #   dependencies required to build them. We use ag-pack to
    #   render the dockerfile, and subsequently build the images.
    #   Optionally, the user may define multiple runtime environments
    #   within a single config.yaml file.
    #   - config.yaml ("images") -> Dockerfile.<basedir> with patched
    #  `add` and `env` -> Build docker image <basedir>:<targe>
    #   - config.yaml ("images") -> docker-compose.<basedir>.yml with
    #  parametrized imag name, http_port, etc. -> check_call("docker-compose up")
    #
    # 2. models:
    #   This section defines the models to serve, and the correspoding
    #   runtime environment to use. Models are defined with a `model_path`,
    #   `model_cls` and default method to dynamically import the relevant
    #   module in Python. The user may define multiple models
    #   within a single config.yaml file.
    #   config.yaml ("models") -> check if ModelSpec can be created from
    #   the config locally; registry happens in the server/container.

    # Check if the config file exists
    path = Path(config_filename)
    if not path.exists():
        raise FileNotFoundError(f"File {config_filename} not found.")

    # If the current directory is not the same as the config file,
    # then raise an error
    if path.absolute().parent != Path.cwd():
        raise ValueError(
            f"Please run this command from the same directory as the config file. "
            f"Current directory: {Path.cwd()}, config file: {config_filename}"
        )

    # Check if docker / docker compose is installed
    if not has_docker():
        raise RuntimeError("Docker is not installed, please set up docker first before serving.")
    docker_compose_cmd = docker_compose_command()
    if not docker_compose_cmd:
        raise RuntimeError("Docker compose is not installed, please set up docker compose first before serving.")

    # Use the directory name as the default docker sandbox name `<sandbox_name>:<target>``
    SANDBOX_DIR = "/app/serve"
    sandbox_name: str = path.absolute().parent.name
    container_sandbox_path: Path = Path(SANDBOX_DIR) / sandbox_name
    logger.debug(f"[config={config_filename}, sandbox={sandbox_name}, container_sandbox={container_sandbox_path}]")

    # Get all the "models" defined in the config
    # Hub.register_from_yaml(config_filename)

    # Load the "images" defined in the config
    config = AGIPackConfig.load_yaml(config_filename)

    # Add the current working directory to the config `add`
    # and include the NOC_HUB_CATALOG_PATH environment variable
    # to the config `env`.
    # Note (spillai): The current working directory is added to /app/serve/<basedir>
    config_basename: Path = Path(config_filename).name
    container_config_path: Path = container_sandbox_path / config_basename
    for _target, image_config in config.images.items():
        image_config.workdir = str(container_sandbox_path)
        image_config.add.append(f"./:{str(container_sandbox_path)}")
        image_config.env["NOS_HUB_CATALOG_PATH"] = f"$NOS_HUB_CATALOG_PATH:{str(container_config_path)}"

    # Render the dockerfiles
    builder = AGIPack(config)
    dockerfiles = builder.render(
        filename=f"Dockerfile.{sandbox_name}", env="prod" if prod else "dev", skip_base_builds=True
    )

    # Check if several targets are defined, if so, expect the user
    # to specify which one to build / serve
    if len(dockerfiles) > 1 and target is None:
        raise ValueError(
            f"Several targets defined in the config, please specify which one to "
            f"build / serve using the `--target` flag. "
            f"Available targets: {', '.join(dockerfiles.keys())}"
        )
    if target is None:
        target = list(dockerfiles.keys())[0]
        logger.debug(f"Using target={target}.")

    # Build the runtime environments (unless target is specified)
    tag_name = None
    for docker_target, filename in dockerfiles.items():
        # Skip if the target is not the one we want to build
        if target is not None and docker_target != target:
            continue
        image_config = config.images[docker_target]
        tag_name = tag.format(name=sandbox_name, target=docker_target)
        cmd = f"docker build -f {filename} --target {docker_target} -t {tag_name} ."

        # Print the command to build the Dockerfile
        tree = Tree(f"📦 [bold white]{docker_target}[/bold white]")
        tree.add(
            f"[bold green]✓[/bold green] Successfully generated Dockerfile (target=[bold white]{docker_target}[/bold white], filename=[bold white]{filename}[/bold white])."
        ).add(f"[green]`{cmd}`[/green]")
        print(tree)
        with redirect_stdout_to_logger(level="DEBUG"):
            builder.build(filename=filename, target=docker_target, tags=[tag_name])

    # Check if the image was built
    if tag_name is None:
        raise ValueError(f"Failed to build target={target}, cannot proceed.")

    # Render the docker-compose file using the tag name, http port, etc.
    SERVE_TEMPLATE = Path(__file__).parent / "templates/docker-compose.serve.yml.j2"
    if not SERVE_TEMPLATE.exists():
        raise FileNotFoundError(f"File {SERVE_TEMPLATE} not found.")
    template_env = Environment(loader=FileSystemLoader(searchpath=SERVE_TEMPLATE.parent))
    template = template_env.get_template(SERVE_TEMPLATE.name)
    logger.debug(f"Using template: {SERVE_TEMPLATE.name}")

    # Render the template using ServeOptions
    options = ServeOptions(
        config=str(container_config_path),
        image=tag_name,
        gpu=has_gpu(),
        http=http,
        http_port=http_port,
        http_max_workers=http_max_workers,
        logging_level=logging_level,
        daemon=daemon,
    )
    content = template.render(**asdict(options))
    logger.debug(f"Rendered template content:\n{content}")

    # Write the docker-compose file
    compose_path = Path.cwd() / f"docker-compose.{sandbox_name}.yml"
    with compose_path.open("w") as f:
        f.write(content)
    print(f"[green]✓[/green] Successfully generated docker-compose file (filename=[bold white]{f.name}[/bold white]).")

    # Launch docker compose with the built images
    cmd = f"{docker_compose_cmd} -f {compose_path.name} up"
    if daemon:
        cmd += " -d"
    print(f"[green]✓[/green] Launching docker compose with command: [bold white]{cmd}[/bold white]")
    proc = subprocess.run(cmd, shell=True)
    if proc.returncode != 0:
        logger.error(f"Failed to serve, e={proc.stderr}")
        raise RuntimeError(f"Failed to serve, e={proc.stderr}")
