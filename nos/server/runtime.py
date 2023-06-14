"""gRPC server runtime using docker executor."""
import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import docker
from nos.constants import DEFAULT_GRPC_PORT  # noqa F401
from nos.logging import LOGGING_LEVEL, logger
from nos.protoc import import_module
from nos.server.docker import DockerRuntime


nos_service_pb2 = import_module("nos_service_pb2")
nos_service_pb2_grpc = import_module("nos_service_pb2_grpc")


NOS_DOCKER_IMAGE_CPU = "autonomi/nos:latest-cpu"
NOS_DOCKER_IMAGE_GPU = "autonomi/nos:latest-gpu"

NOS_INFERENCE_SERVICE_CONTAINER_NAME = "nos-inference-service"
NOS_INFERENCE_SERVICE_CMD = "nos-grpc-server"


@dataclass
class InferenceServiceRuntimeConfig:
    """Inference service configuration."""

    image: str
    """Docker image."""

    name: str = NOS_INFERENCE_SERVICE_CONTAINER_NAME
    """Container name (unique)."""

    command: Union[str, List[str]] = field(default_factory=lambda: [NOS_INFERENCE_SERVICE_CMD])
    """Command to run."""

    ports: Dict[int, int] = field(default_factory=lambda: {DEFAULT_GRPC_PORT: DEFAULT_GRPC_PORT})
    """Ports to expose."""

    environment: Dict[str, str] = field(default_factory=lambda: {"NOS_LOGGING_LEVEL": LOGGING_LEVEL})
    """Environment variables."""

    volumes: Dict[str, Dict[str, str]] = field(
        default_factory=lambda: {
            str(Path.home() / ".nosd"): {"bind": "/app/.nos", "mode": "rw"},
        }
    )
    """Volumes to mount."""

    shm_size: str = "4g"
    """Size of /dev/shm."""

    detach: bool = True
    """Whether to run the container in detached mode."""

    gpu: bool = False
    """Whether to start the container with GPU support."""

    kwargs: Dict[str, Any] = field(
        default_factory=lambda: {
            "nano_cpus": int(6e9),
            "mem_limit": "6g",
        }
    )
    """Additional keyword-arguments to pass to `DockerRuntime.start`."""


class InferenceServiceRuntime:
    """Inference service runtime.

    This class is responsible for handling the lifecycle of the
    inference service docker runtime.

    Parameters:
        cfg (InferenceServiceConfig): Inference service configuration.
    """

    configs = {
        "cpu": InferenceServiceRuntimeConfig(
            image=NOS_DOCKER_IMAGE_CPU,
            name=f"{NOS_INFERENCE_SERVICE_CONTAINER_NAME}-cpu",
            gpu=False,
        ),
        "gpu": InferenceServiceRuntimeConfig(
            image=NOS_DOCKER_IMAGE_GPU,
            name=f"{NOS_INFERENCE_SERVICE_CONTAINER_NAME}-gpu",
            gpu=True,
        ),
        "mmdet-dev": InferenceServiceRuntimeConfig(
            image="autonomi/nos:latest-mmdet-dev",
            name=f"{NOS_INFERENCE_SERVICE_CONTAINER_NAME}-mmdet",
            gpu=True,
            environment={
                "NOS_LOGGING_LEVEL": LOGGING_LEVEL,
                "NOS_ENV": "mmdet-dev",
            },
        ),
    }

    def __init__(self, runtime: str = "cpu", name: str = NOS_INFERENCE_SERVICE_CONTAINER_NAME):
        """Initialize the inference runtime.

        Args:
            runtime (str, optional): Inference runtime. Defaults to "cpu".
            name (str, optional): Inference runtime name. Defaults to "nos-inference-service".
        """
        if runtime not in self.configs:
            raise ValueError(f"Invalid inference runtime: {runtime}, available: {list(self.configs.keys())}")
        self.cfg = copy.deepcopy(self.configs[runtime])
        self.cfg.name = name

        self._runtime = DockerRuntime.get()

    def __repr__(self) -> str:
        return f"InferenceServiceRuntime(image={self.cfg.image}, name={self.cfg.name}, gpu={self.cfg.gpu})"

    @classmethod
    def list(self, **kwargs) -> List[docker.models.containers.Container]:
        """List running docker containers."""
        containers = DockerRuntime.get().list(**kwargs)
        return [
            container for container in containers if container.name.startswith(NOS_INFERENCE_SERVICE_CONTAINER_NAME)
        ]

    def start(self, **kwargs) -> docker.models.containers.Container:
        """Start the inference runtime.

        Args:
            **kwargs: Additional keyword-arguments to pass to `DockerRuntime.start`.
        """
        logger.info(f"Starting inference runtime with image: {self.cfg.image}")

        # Override config with supplied kwargs
        for k in list(kwargs.keys()):
            value = kwargs[k]
            if hasattr(self.cfg, k):
                setattr(self.cfg, k, value)
            else:
                self.cfg.kwargs[k] = value
                logger.debug(f"Overriding inference runtime config: {k}={value}")

        # Start inference runtime
        container = self._runtime.start(
            image=self.cfg.image,
            name=self.cfg.name,
            command=self.cfg.command,
            ports=self.cfg.ports,
            environment=self.cfg.environment,
            volumes=self.cfg.volumes,
            detach=self.cfg.detach,
            gpu=self.cfg.gpu,
            **self.cfg.kwargs,
        )
        logger.info(f"Started inference runtime: {self}")
        return container

    def stop(self, timeout: int = 30) -> docker.models.containers.Container:
        return self._runtime.stop(self.cfg.name, timeout=timeout)

    def get_container(self) -> docker.models.containers.Container:
        return self._runtime.get_container(self.cfg.name)

    def get_container_name(self) -> Optional[str]:
        return self._runtime.get_container(self.cfg.name).name

    def get_container_id(self) -> Optional[str]:
        return self._runtime.get_container_id(self.cfg.name)

    def get_container_status(self) -> Optional[str]:
        return self._runtime.get_container_status(self.cfg.name)

    def get_container_logs(self, **kwargs) -> Iterable[str]:
        return self._runtime.get_container_logs(self.cfg.name, **kwargs)
