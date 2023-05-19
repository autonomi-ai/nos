"""Ray executor for NOS.

This module provides a Ray executor for NOS. The Ray executor is a singleton
instance that can be used to start a Ray head, connect to an existing Ray
cluster, and submit tasks to Ray. We use Ray as a backend for distributed
computation and containerize them in docker to isolate the environment.
"""
import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

import psutil
import ray
import rich.console
import rich.panel
import rich.status

from nos.logging import LOGGING_LEVEL


logger = logging.getLogger(__name__)

NOS_RAY_NS = os.getenv("NOS_RAY_NS", "nos-dev")
NOS_RAY_RUNTIME_ENV = os.getenv("NOS_RAY_ENV", None)


@dataclass
class RayRuntimeSpec:
    namespace: str = NOS_RAY_NS
    """Namespace for Ray runtime."""
    runtime_env: str = NOS_RAY_RUNTIME_ENV
    """Runtime environment for Ray runtime."""


@dataclass
class RayExecutor:
    """Executor for Ray."""

    _instance: "RayExecutor" = None
    """Singleton instance of RayExecutor."""
    spec: RayRuntimeSpec = RayRuntimeSpec()
    """Runtime spec for Ray."""

    @classmethod
    def get(cls) -> "RayExecutor":
        """Get the singleton instance of RayExecutor."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def is_initialized(self) -> bool:
        """Check if Ray is initialized."""
        return ray.is_initialized()

    def init(self, max_attempts: int = 5, timeout: int = 60, retry_interval: int = 5) -> None:
        """Initialize Ray exector.

        This implementation forces Ray to start a new cluster instance via
        `ray.init(address="local")` and then connecting to the
        server via `ray.init(address="auto")`. The first call to
        `ray.init(address="auto")` raises a `ConnectionError` and proceeds to
        force-start a new ray cluster instance, followed by a second call to
        `ray.init(address="auto")` which successfully connects to the server.

        In the case of a Ray cluster already running, the first call to
        `ray.init(address="auto")` will successfully connect to the server.

        Args:
            max_attempts: Number of retries to attempt to connect to an existing
            timeout: Time to wait for Ray to start. Defaults to 60 seconds.
            retry_interval: Time to wait between retries. Defaults to 5 seconds.
        """
        level = getattr(logging, LOGGING_LEVEL)

        # Ignore predefined RAY_ADDRESS environment variable.
        if "RAY_ADDRESS" in os.environ:
            del os.environ["RAY_ADDRESS"]

        st = time.time()
        attempt = 0

        # Attempt to connect to an existing ray cluster.
        # Allow upto 5 attempts, or if timeout of 60 seconds is reached.
        console = rich.console.Console()
        while time.time() - st <= timeout and attempt < max_attempts:
            # Attempt to connect to an existing ray cluster in the background.
            try:
                with console.status(
                    "[bold green] InferenceExecutor :: Connecting to backend ... [/bold green]"
                ) as status:
                    ray.init(
                        address="auto",
                        namespace=self.spec.namespace,
                        runtime_env=self.spec.runtime_env,
                        ignore_reinit_error=True,
                        configure_logging=True,
                        logging_level=logging.ERROR,
                        log_to_driver=level <= logging.ERROR,
                    )
                    status.stop()
                    console.print("[bold green] ✓ InferenceExecutor :: Connected to backend. [/bold green]")
                return True
            except ConnectionError as exc:
                # If Ray head is not running (this results in a ConnectionError),
                # start it in a background subprocess.
                if attempt > 0:
                    logger.error(
                        f"Failed to connect to InferenceExecutor.\n"
                        f"{exc}\n"
                        f"Retrying {attempt}/{max_attempts} after {retry_interval}s..."
                    )
                self.start()
                attempt += 1

                time.sleep(retry_interval)
        return False

    def start(self) -> None:
        """Force-start a local instance of Ray head."""
        level = getattr(logging, LOGGING_LEVEL)

        console = rich.console.Console()
        with console.status(
            "[bold green] InferenceExecutor :: Backend initializing (as daemon) ... [/bold green]"
        ) as status:
            try:
                ray.init(
                    _node_name="nos-executor",
                    address="local",
                    namespace=self.spec.namespace,
                    runtime_env=self.spec.runtime_env,
                    ignore_reinit_error=False,
                    include_dashboard=False,
                    configure_logging=True,
                    logging_level=logging.ERROR,
                    log_to_driver=level <= logging.ERROR,
                )
                status.stop()
            except ConnectionError as exc:
                raise RuntimeError(f"Failed to start executor: exc={exc}.")
            console.print("[bold green] ✓ InferenceExecutor :: Backend initialized. [/bold green]")

    def stop(self) -> None:
        """Stop Ray head."""
        console = rich.console.Console()
        with console.status("[bold green] InferenceExecutor :: Backend stopping ... [/bold green]"):
            try:
                ray.shutdown()
            except Exception as exc:
                raise RuntimeError(f"Failed to stop executor: exc={exc}.")
            console.print("[bold green] ✓ InferenceExecutor :: Backend stopped. [/bold green]")

    @property
    def pid(self) -> Optional[int]:
        """Get PID of Ray head."""
        for proc in psutil.process_iter(attrs=["pid", "name"]):
            if proc.name() == "raylet":
                return proc.pid
        return None


def init(*args, **kwargs) -> bool:
    """Initialize Ray executor."""
    exector = RayExecutor.get()
    return exector.init(*args, **kwargs)
