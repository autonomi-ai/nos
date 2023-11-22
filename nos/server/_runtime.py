"""gRPC server runtime using docker executor."""
import copy
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import psutil

import docker
from docker.types import LogConfig
from nos.common.shm import NOS_SHM_ENABLED
from nos.constants import (  # noqa F401
    DEFAULT_GRPC_PORT,
    NOS_MEMRAY_ENABLED,
    NOS_PROFILING_ENABLED,
    NOS_RAY_DASHBOARD_ENABLED,
)
from nos.logging import LOGGING_LEVEL, logger
from nos.protoc import import_module
from nos.version import __version__

from ._docker import DockerRuntime


nos_service_pb2 = import_module("nos_service_pb2")
nos_service_pb2_grpc = import_module("nos_service_pb2_grpc")


NOS_DOCKER_IMAGE_CPU = f"autonomi/nos:{__version__}-cpu"
NOS_DOCKER_IMAGE_GPU = f"autonomi/nos:{__version__}-gpu"

NOS_INFERENCE_SERVICE_CONTAINER_NAME = "nos-inference-service"
NOS_INFERENCE_SERVICE_CMD = ["/app/entrypoint.sh"]  # this needs to be consistent with the Dockerfile

NOS_SUPPORTED_BACKENDS = ("cpu", "cuda", "mps", "neuron")


def _default_environment(kwargs: Dict[str, str] = None) -> Dict[str, str]:
    """Default environment variables that can be overriden for the
    `InferenceServiceRuntimeConfig` default_factory."""
    environment = {
        "OMP_NUM_THREADS": psutil.cpu_count(logical=False),
        "HUGGINGFACE_HUB_TOKEN": os.environ.get("HUGGINGFACE_HUB_TOKEN", ""),
        "NOS_LOGGING_LEVEL": LOGGING_LEVEL,
        "NOS_PROFILING_ENABLED": int(NOS_PROFILING_ENABLED),
        "NOS_SHM_ENABLED": int(NOS_SHM_ENABLED),
        "NOS_MEMRAY_ENABLED": int(NOS_MEMRAY_ENABLED),
        "NOS_RAY_DASHBOARD_ENABLED": int(NOS_RAY_DASHBOARD_ENABLED),
    }
    if kwargs is not None:
        environment.update(kwargs)
    return environment


def _default_volume(mounts: Dict[str, Dict[str, str]] = None) -> Dict[str, Dict[str, str]]:
    """Default volumes that can be overriden for the
    `InferenceServiceRuntimeConfig` default_factory."""
    environment = {
        str(Path.home() / ".nosd"): {"bind": "/app/.nos", "mode": "rw"},  # nos cache
        "/dev/shm": {"bind": "/dev/shm", "mode": "rw"},  # shared-memory transport
    }
    if mounts is not None:
        environment.update(mounts)
    return environment


@dataclass
class InferenceServiceRuntimeConfig:
    """Inference service configuration."""

    image: str
    """Docker image."""

    name: str = NOS_INFERENCE_SERVICE_CONTAINER_NAME
    """Container name (unique)."""

    command: Union[str, List[str]] = field(default_factory=lambda: NOS_INFERENCE_SERVICE_CMD)
    """Command to run."""

    ports: Dict[int, int] = field(default_factory=lambda: {DEFAULT_GRPC_PORT: DEFAULT_GRPC_PORT})
    """Ports to expose."""

    environment: Dict[str, str] = field(default_factory=lambda: _default_environment())
    """Environment variables."""

    volumes: Dict[str, Dict[str, str]] = field(default_factory=lambda: _default_volume())
    """Volumes to mount."""

    shm_size: str = "4g"
    """Size of /dev/shm."""

    ipc_mode: str = "host"
    """IPC mode."""

    detach: bool = True
    """Whether to run the container in detached mode."""

    device: str = None
    """Device to request (i.e. gpu, inf2). Defaults to None (i.e. cpu)."""

    kwargs: Dict[str, Any] = field(
        default_factory=lambda: {
            "nano_cpus": int(6e9),
            "mem_limit": "6g",
            "log_config": {"type": LogConfig.types.JSON, "config": {"max-size": "100m", "max-file": "10"}},
        }
    )
    """Additional keyword-arguments to pass to `DockerRuntime.start`."""


class InferenceServiceRuntime:
    """Inference service runtime.

    This class is responsible for handling the lifecycle of the
    inference service docker runtime.

    Attributes:
        configs (InferenceServiceConfig): Inference service configuration.
    """

    configs = {
        "cpu": InferenceServiceRuntimeConfig(
            image=NOS_DOCKER_IMAGE_CPU,
            name=f"{NOS_INFERENCE_SERVICE_CONTAINER_NAME}-cpu",
            kwargs={
                "nano_cpus": int(6e9),
                "mem_limit": "6g",
                "log_config": {"type": LogConfig.types.JSON, "config": {"max-size": "100m", "max-file": "10"}},
            },
        ),
        "gpu": InferenceServiceRuntimeConfig(
            image=NOS_DOCKER_IMAGE_GPU,
            name=f"{NOS_INFERENCE_SERVICE_CONTAINER_NAME}-gpu",
            device="gpu",
            kwargs={
                "nano_cpus": int(8e9),
                "mem_limit": "12g",
                "log_config": {"type": LogConfig.types.JSON, "config": {"max-size": "100m", "max-file": "10"}},
            },
        ),
        "trt": InferenceServiceRuntimeConfig(
            image="autonomi/nos:latest-trt",
            name=f"{NOS_INFERENCE_SERVICE_CONTAINER_NAME}-trt",
            device="gpu",
            kwargs={
                "nano_cpus": int(8e9),
                "mem_limit": "12g",
                "log_config": {"type": LogConfig.types.JSON, "config": {"max-size": "100m", "max-file": "10"}},
            },
        ),
        "inf2": InferenceServiceRuntimeConfig(
            image="autonomi/nos:latest-inf2",
            name=f"{NOS_INFERENCE_SERVICE_CONTAINER_NAME}-inf2",
            device="inf2",
            environment=_default_environment({"NEURON_RT_VISIBLE_CORES": 2}),
            kwargs={
                "nano_cpus": int(8e9),
                "log_config": {"type": LogConfig.types.JSON, "config": {"max-size": "100m", "max-file": "10"}},
            },
        ),
    }

    def __init__(self, runtime: str = "cpu", name: str = None):
        """Initialize the inference runtime.

        Args:
            runtime (str, optional): Inference runtime. Defaults to "cpu".
            name (str, optional): Inference runtime name. Defaults to "nos-inference-service".
        """
        if runtime not in self.configs:
            raise ValueError(f"Invalid inference runtime: {runtime}, available: {list(self.configs.keys())}")
        self.cfg = copy.deepcopy(self.configs[runtime])
        if name is not None:
            self.cfg.name = name

        self._runtime = DockerRuntime.get()

    def __repr__(self) -> str:
        return f"InferenceServiceRuntime(image={self.cfg.image}, name={self.cfg.name}, device={self.cfg.device})"

    @staticmethod
    def detect() -> str:
        """Auto-detect inference runtime."""
        from nos.common.system import has_gpu, is_aws_inf2

        if is_aws_inf2():
            return "inf2"
        elif has_gpu():
            return "gpu"
        else:
            return "cpu"

    @staticmethod
    def list(**kwargs) -> List[docker.models.containers.Container]:
        """List running docker containers."""
        containers = DockerRuntime.get().list(**kwargs)
        return [
            container for container in containers if container.name.startswith(NOS_INFERENCE_SERVICE_CONTAINER_NAME)
        ]

    @classmethod
    def supported_runtimes(cls) -> List[str]:
        """Get supported runtimes."""
        return list(cls.configs.keys())

    def start(self, **kwargs: Any) -> docker.models.containers.Container:
        """Start the inference runtime.

        Args:
            **kwargs: Additional keyword-arguments to pass to `DockerRuntime.start`.
        """
        logger.debug(f"Starting inference runtime with image: {self.cfg.image}")

        # Override dict values
        for k in ("ports", "volumes", "environment"):
            if k in kwargs:
                self.cfg.__dict__[k].update(kwargs.pop(k))
                logger.debug(f"Updating runtime configuration [key={k}, value={self.cfg.__dict__[k]}]")

        # Override config with supplied kwargs
        for k in list(kwargs.keys()):
            value = kwargs[k]
            if hasattr(self.cfg, k):
                setattr(self.cfg, k, value)
                logger.debug(f"Overriding inference runtime config: {k}={value}")
            else:
                self.cfg.kwargs[k] = value

        # Start inference runtime
        container = self._runtime.start(
            image=self.cfg.image,
            name=self.cfg.name,
            command=self.cfg.command,
            ports=self.cfg.ports,
            environment=self.cfg.environment,
            volumes=self.cfg.volumes,
            detach=self.cfg.detach,
            device=self.cfg.device,
            ipc_mode=self.cfg.ipc_mode,
            **self.cfg.kwargs,
        )
        logger.debug(f"Started inference runtime: {self}")
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
