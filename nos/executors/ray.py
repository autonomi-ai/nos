"""Ray executor for NOS.

This module provides a Ray executor for NOS. The Ray executor is a singleton
instance that can be used to start a Ray head, connect to an existing Ray
cluster, and submit tasks to Ray. We use Ray as a backend for distributed
computation and containerize them in docker to isolate the environment.
"""
import logging
import os
import time
from dataclasses import dataclass, field
from functools import cached_property
from typing import List, Optional

import psutil
import ray
import rich.console
import rich.panel
import rich.status
from ray.job_submission import JobSubmissionClient

from nos.common.metaclass import SingletonMetaclass
from nos.constants import (
    NOS_RAY_DASHBOARD_ENABLED,
    NOS_RAY_ENV,
    NOS_RAY_JOB_CLIENT_ADDRESS,
    NOS_RAY_NS,
    NOS_RAY_OBJECT_STORE_MEMORY,
)
from nos.logging import LOGGING_LEVEL


logger = logging.getLogger(__name__)
logging.getLogger("ray").setLevel(logging.ERROR)


@dataclass
class RayRuntimeSpec:
    namespace: str = NOS_RAY_NS
    """Namespace for Ray runtime."""
    runtime_env: str = NOS_RAY_ENV
    """Runtime environment for Ray runtime."""


@dataclass
class RayExecutor(metaclass=SingletonMetaclass):
    """Executor for Ray."""

    spec: RayRuntimeSpec = RayRuntimeSpec()
    """Runtime spec for Ray."""

    @classmethod
    def get(cls) -> "RayExecutor":
        """Get the singleton instance of RayExecutor."""
        return cls()

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
                    logger.debug(f"Connecting to executor: namespace={self.spec.namespace}")
                    ray.init(
                        address="auto",
                        namespace=self.spec.namespace,
                        ignore_reinit_error=True,
                        logging_level="error",
                    )
                    status.stop()
                    console.print("[bold green] ✓ InferenceExecutor :: Connected to backend. [/bold green]")
                    logger.debug(
                        f"Connected to executor: namespace={self.spec.namespace} (time={time.time() - st:.2f}s)"
                    )
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
                    time.sleep(retry_interval)
                else:
                    logger.debug("No executor found, starting a new one")
                    self.start()
                attempt += 1
                continue
        logger.error(f"Failed to connect to InferenceExecutor: namespace={self.spec.namespace}.")
        return False

    def start(self) -> None:
        """Force-start a local instance of Ray head."""
        level = getattr(logging, LOGGING_LEVEL)

        start_t = time.time()
        console = rich.console.Console()
        console.print("[bold green] ✓ InferenceExecutor :: Backend initializing (as daemon) ... [/bold green]")
        try:
            logger.debug(f"Starting executor: namespace={self.spec.namespace}")
            ray.init(
                _node_name="nos-executor",
                address="local",
                namespace=self.spec.namespace,
                object_store_memory=NOS_RAY_OBJECT_STORE_MEMORY,
                ignore_reinit_error=False,
                include_dashboard=NOS_RAY_DASHBOARD_ENABLED,
                configure_logging=True,
                logging_level=logging.ERROR,
                log_to_driver=level <= logging.ERROR,
                dashboard_host="0.0.0.0" if NOS_RAY_DASHBOARD_ENABLED else None,
            )
            logger.debug(f"Started executor: namespace={self.spec.namespace} (time={time.time() - start_t:.2f}s)")
        except ConnectionError as exc:
            logger.error(f"Failed to start executor: exc={exc}.")
            raise RuntimeError(f"Failed to start executor: exc={exc}.")
        console.print(
            f"[bold green] ✓ InferenceExecutor :: Backend initialized (elapsed={time.time() - start_t:.1f}s). [/bold green]"
        )
        logger.debug(f"Started executor: namespace={self.spec.namespace} (time={time.time() - start_t}s)")

    def stop(self) -> None:
        """Stop Ray head."""
        console = rich.console.Console()
        console.print("[bold green] InferenceExecutor :: Backend stopping ... [/bold green]")
        try:
            logger.debug(f"Stopping executor: namespace={self.spec.namespace}")
            ray.shutdown()
            logger.debug(f"Stopped executor: namespace={self.spec.namespace}")
        except Exception as exc:
            logger.error(f"Failed to stop executor: exc={exc}.")
            raise RuntimeError(f"Failed to stop executor: exc={exc}.")
        console.print("[bold green] ✓ InferenceExecutor :: Backend stopped. [/bold green]")

    @property
    def pid(self) -> Optional[int]:
        """Get PID of Ray head."""
        for proc in psutil.process_iter(attrs=["pid", "name"]):
            if proc.name() == "raylet":
                return proc.pid
        return None

    @cached_property
    def jobs(self) -> "RayJobExecutor":
        """Get the ray jobs executor."""
        return RayJobExecutor()


@dataclass
class RayJobExecutor(metaclass=SingletonMetaclass):
    """Ray job executor."""

    client: JobSubmissionClient = field(init=False)
    """Job submission client."""

    def __post_init__(self):
        """Post-initialization."""
        if not ray.is_initialized():
            raise RuntimeError("Ray executor is not initialized.")
        self.client = JobSubmissionClient(NOS_RAY_JOB_CLIENT_ADDRESS)

    def submit(self, *args, **kwargs) -> str:
        """Submit a job to Ray."""
        job_id = self.client.submit_job(*args, **kwargs)
        logger.debug(f"Submitted job with id: {job_id}")
        return job_id

    def list(self) -> List[str]:
        """List all jobs."""
        return self.client.list_jobs()

    def info(self, job_id: str) -> str:
        """Get info for a job."""
        return self.client.get_job_info(job_id)

    def status(self, job_id: str) -> str:
        """Get status for a job."""
        return self.client.get_job_status(job_id)

    def logs(self, job_id: str) -> str:
        """Get logs for a job."""
        return self.client.get_job_logs(job_id)

    def wait(self, job_id: str, timeout: int = 600, retry_interval: int = 5) -> str:
        """Wait for a job to complete."""
        status = None
        st = time.time()
        while time.time() - st < timeout:
            status = self.status(job_id)
            if str(status) == "SUCCEEDED":
                logger.debug(f"Training job completed [job_id={job_id}, status={status}]")
                return status
            else:
                logger.debug(f"Training job not completed yet [job_id={job_id}, status={status}]")
                time.sleep(retry_interval)
        logger.warning(f"Training job timed out [job_id={job_id}, status={status}]")
        return status


def init(*args, **kwargs) -> bool:
    """Initialize Ray executor."""
    logger.debug(f"Initializing executor: args={args}, kwargs={kwargs}")
    exector = RayExecutor.get()
    return exector.init(*args, **kwargs)
