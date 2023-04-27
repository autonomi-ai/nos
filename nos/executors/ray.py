"""Ray executor for NOS.

This module provides a Ray executor for NOS. The Ray executor is a singleton
instance that can be used to start a Ray head, connect to an existing Ray
cluster, and submit tasks to Ray. We use Ray as a backend for distributed
computation and containerize them in docker to isolate the environment.
"""
import logging
import os
import subprocess
import time
from dataclasses import dataclass
from typing import Optional

import psutil
import ray
import rich.console
import rich.panel
import rich.status

from nos.constants import NOS_TMP_DIR
from nos.logging import LOGGING_LEVEL


logger = logging.getLogger(__name__)

NOS_RAY_ADDRESS = os.environ.get("RAY_ADDRESS", "auto")
NOS_RAY_NS = os.getenv("NOS_RAY_NS", "nos-dev")
NOS_RAY_RUNTIME_ENV = os.getenv("NOS_RAY_ENV", None)


@dataclass
class RayRuntimeSpec:
    address: str = NOS_RAY_ADDRESS
    """Address of Ray head."""
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
        """Initialize Ray exector (as a daemon).

        Currently, this method will first attempt to connect to an existing
        Ray cluster (address="auto"). If it fails, it will start a Ray head
        as a daemon, attempt to re-connect (upto a maximum of 5
        attempts, or if timeout of 60 seconds is reached).

        Args:
            max_attempts: Number of retries to attempt to connect to an existing
            timeout: Time to wait for Ray to start. Defaults to 60 seconds.
            retry_interval: Time to wait between retries. Defaults to 5 seconds.
        """
        level = getattr(logging, LOGGING_LEVEL)

        st = time.time()
        attempt = 0

        # Attempt to connect to an existing ray cluster.
        # Allow upto 5 attempts, or if timeout of 60 seconds is reached.
        console = rich.console.Console()
        while time.time() - st <= timeout and attempt < max_attempts:
            # Attempt to connect to an existing ray cluster in the background.
            try:
                with console.status("[bold green] Connecting to ray cluster ... [/bold green]"):
                    ray.init(
                        address=self.spec.address,
                        namespace=self.spec.namespace,
                        runtime_env=self.spec.runtime_env,
                        ignore_reinit_error=True,
                        configure_logging=True,
                        logging_level=LOGGING_LEVEL,
                        log_to_driver=level <= logging.INFO,
                    )
                    console.print(f"[bold green] ✓ Connected to ray cluster: {self.spec.address}[/bold green]")
                return True
            except ConnectionError:
                # If Ray head is not running (this results in a ConnectionError),
                # start it in a background subprocess.
                self.start()
                attempt += 1
                logger.info(
                    f"Failed to connect to Ray head. Retrying {attempt}/{max_attempts} after {retry_interval}s..."
                )
                time.sleep(retry_interval)
        return False

    def start(self, wait: int = 5) -> Optional[int]:
        """Start Ray head as daemon.

        Args:
            wait: Time to wait for Ray to start. Defaults to 5 seconds.
        """
        console = rich.console.Console()
        with console.status("[bold green] Starting ray head (as daemon) ... [/bold green]"):
            # Check if Ray head is already running
            if self.pid:
                console.print("[bold yellow] Ray head is already running. [/bold yellow]")
                return self.pid
            # Start Ray head if not running
            RAY_TMP_DIR = NOS_TMP_DIR / "ray"
            RAY_TMP_DIR.mkdir(parents=True, exist_ok=True)
            cmd = f"ray start --head --storage {RAY_TMP_DIR}"
            proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            time.sleep(wait)
            if proc.poll() != 0:
                raise RuntimeError("Failed to start Ray.")
            return self.pid

    def stop(self, wait: int = 5) -> Optional[int]:
        """Stop Ray head."""
        console = rich.console.Console()
        with console.status("[bold green] Stopping ray head... [/bold green]"):
            # Check if Ray head is running
            if not self.pid:
                console.print("[bold yellow] Ray head is not running.[/bold yellow]")
                return
            # Stop Ray head if pid is valid
            cmd = "ray stop -f"
            proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            time.sleep(wait)
            if proc.poll() != 0:
                raise RuntimeError("Failed to stop Ray.")
            return self.pid

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
