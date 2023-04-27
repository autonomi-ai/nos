import logging
import os
import re
import sys
import time
import traceback
from typing import Any, Dict

import ray
from ray import serve
from ray.serve._private import api as _private_api
from ray.serve.exceptions import RayServeException
from ray.serve.handle import RayServeDeploymentHandle
from rich.console import Console

from nos.executors.ray import RayExecutor
from nos.hub import ModelSpec
from nos.serve.ingress import APIIngress


logger = logging.getLogger(__name__)
console = Console()


NOS_SERVE_DEFAULT_HTTP_HOST = "127.0.0.1"
NOS_SERVE_DEFAULT_HTTP_PORT = 6169


def get_deployment_name(model_name: str = None) -> str:
    """Get the deployment name from the model name."""
    return re.sub(r"[^a-zA-Z0-9_]", "_", model_name)


def list():
    """List all deployments."""
    try:
        return serve.list_deployments()
    except RayServeException:
        print("Ray Serve is not running.")
        return None


def deployment(
    model_name: str,
    model_spec: ModelSpec,
    deployment_config: Dict[str, Dict[str, Any]] = None,
    host: str = NOS_SERVE_DEFAULT_HTTP_HOST,
    port: int = NOS_SERVE_DEFAULT_HTTP_PORT,
    daemon: bool = False,
) -> Any:
    """Serve deployment wrapper for NOS.

    This wrapper is used to create a serve deployment from a model handle.
    The model handle is a Ray actor handle that can be used to call the model
    remotely. The serve deployment is created using the `serve.deployment`
    decorator.

    The `nos.serve.deployment` does 2 things:
        1. Create a serve deployment from the model_handle.

            deployment = serve.deployment(
                ray_actor_options=...,
                autoscaling_config=...
            )(model_handle)

        2. Create an API-ingress serve deployment using the
           deployment handle created above.

            @serve.deployment(...)
            @serve.ingress(app)
            class APIIngress:
                def __init__(self, model_handle) -> None:
                    ...

                @app.get("/health", status_code=status.HTTP_200_OK)
                async def _health(self):
                    return {"status": "ok"}

                @app.post("/predict")
                async def generate(self, request: PredictionRequest):
                    ...

           entrypoint = APIIngress.bind(deployment.bind())

    """
    # Create the serve deployment from the model handle
    model_cls = model_spec.cls
    deployment = serve.deployment(**deployment_config)(model_cls)
    deployment_handle: RayServeDeploymentHandle = deployment.bind(*model_spec.args, **model_spec.kwargs)

    # Bind the deployment to the API ingress
    entrypoint = APIIngress.bind(deployment_handle)

    # Run the API ingress (optionally as a daemon)
    deployment_name = get_deployment_name(model_name)

    # Setting the runtime_env here will set defaults for the deployments.
    # TODO (spillai): Add support for custom runtime-env and working-dir
    executor = RayExecutor.get()
    executor.init()

    # Start the Ray Serve instance (in detached mode)
    with console.status("[bold green] Starting ray serve ... [/bold green]"):
        _private_api.serve_start(
            detached=True,
            http_options={"host": host, "port": port, "location": "EveryNode"},
        )
        console.print("[bold green] ✓ Ray serve started. [/bold green]")

    # Start the Ray Serve instance (in detached mode)
    with console.status(f"[bold green] Deploying \[name={model_name}] [/bold green]") as status:  # noqa
        # Run the deployment (optionally as a daemon)
        try:
            serve.run(entrypoint, host=host, port=port, name=deployment_name)
            console.print(
                f"[bold green] ✓ Deployment complete. \[address=http://{host}:{port}, name={model_name}, id={deployment_name}] [/bold green]",  # noqa
            )
            status.stop()
            if not daemon:
                while True:
                    time.sleep(10)

        # Graceful shutdown with KeyboardInterrupt
        except KeyboardInterrupt:
            console.print("[yellow] KeyboardInterrupt, shutting down deployment ... [/yellow]")
            serve.shutdown()
            sys.exit()

        # Graceful shutdown with unexpected error
        except Exception:
            traceback.print_exc()
            console.print(
                "[red] ⁉️ Received unexpected error, see console logs for more details. Shutting " "down...[/red]"
            )
            serve.shutdown()
            sys.exit()
