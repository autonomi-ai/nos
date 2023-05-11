"""gRPC server runtime using docker executor."""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from nos.constants import DEFAULT_GRPC_PORT  # noqa F401
from nos.executors.docker import DockerExecutor
from nos.experimental.grpc.protoc import import_module
from nos.logging import LOGGING_LEVEL, logger


nos_service_pb2 = import_module("nos_service_pb2")
nos_service_pb2_grpc = import_module("nos_service_pb2_grpc")


NOS_DOCKER_IMAGE_GPU = "autonomi-ai/nos:latest-gpu"
NOS_DOCKER_IMAGE_CPU = "autonomi-ai/nos:latest-cpu"

NOS_GRPC_SERVER_CONTAINER_NAME = "nos-grpc-server"
NOS_GRPC_SERVER_CMD = "nos-grpc-server"


@dataclass
class InferenceServiceRuntime:
    """Inference service runtime."""

    _executor: DockerExecutor = None
    """Singleton DockerExecutor instance to run inference."""

    def __init__(self):
        """Initialize the runtime."""
        self._executor = DockerExecutor.get()
        logger.info(f"Initialized runtime: {self._executor}")

    def ready(self) -> bool:
        """Check if the inference runtime is ready."""
        status = self.get_status()
        return status and status == "running"

    def id(self) -> Optional[str]:
        """Get the inference runtime container ID."""
        container = self._executor.get_container(NOS_GRPC_SERVER_CONTAINER_NAME)
        return container.id if container else None

    def start(self, detach: bool = True, gpu: bool = False, shm_size: str = "4g", **kwargs):
        """Start the inference runtime.

        Args:
            gpu (bool, optional): Whether to start the runtime with GPU support. Defaults to False.
        """
        image = NOS_DOCKER_IMAGE_GPU if gpu else NOS_DOCKER_IMAGE_CPU
        logger.info(f"Starting inference runtime with image: {image}")
        nos_docker_path = Path.home() / ".nos_docker"
        self._executor.start(
            image=image,
            container_name=NOS_GRPC_SERVER_CONTAINER_NAME,
            command=[NOS_GRPC_SERVER_CMD],
            ports={DEFAULT_GRPC_PORT: DEFAULT_GRPC_PORT},
            environment={
                "NOS_LOGGING_LEVEL": LOGGING_LEVEL,
            },
            volumes={
                str(nos_docker_path): {"bind": "/app/.nos", "mode": "rw"},
                "/tmp/docker": {"bind": "/tmp", "mode": "rw"},
            },
            shm_size=shm_size,
            detach=detach,
            remove=True,
            gpu=gpu,
            **kwargs,
        )
        logger.info(f"Started inference runtime: {self._executor}")

    def stop(self) -> None:
        """Stop the inference runtime."""
        self._executor.stop(NOS_GRPC_SERVER_CONTAINER_NAME)
        logger.info(f"Stopped inference runtime: {self._executor}")

    def get_logs(self) -> Optional[str]:
        """Get the inference runtime logs."""
        return self._executor.get_logs(NOS_GRPC_SERVER_CONTAINER_NAME)

    def get_status(self) -> Optional[str]:
        """Get the inference runtime status."""
        return self._executor.get_container_status(NOS_GRPC_SERVER_CONTAINER_NAME)
