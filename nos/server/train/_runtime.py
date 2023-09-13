from dataclasses import dataclass, field
from typing import Any, Dict, List

import docker
from nos.server._docker import DockerRuntime
from nos.server._runtime import InferenceServiceRuntime, InferenceServiceRuntimeConfig


NOS_CUSTOM_SERVICE_CONTAINER_NAME = "nos-custom-service"


@dataclass
class CustomServiceRuntimeConfig(InferenceServiceRuntimeConfig):
    kwargs: Dict[str, Any] = field(
        default_factory=lambda: {
            "nano_cpus": int(8e9),
            "mem_limit": "12g",
        }
    )
    """Additional keyword-arguments to pass to `DockerRuntime.start`."""


class CustomServiceRuntime(InferenceServiceRuntime):
    """Custom service runtime, inherits from InferenceServiceRuntime."""

    configs = {
        "mmdet-gpu": CustomServiceRuntimeConfig(
            image="autonomi/nos:latest-mmdet-gpu",
            name=f"{NOS_CUSTOM_SERVICE_CONTAINER_NAME}-mmdet-gpu",
            gpu=True,
            ports={
                8265: 8265,
            },
        ),
        "diffusers-gpu": CustomServiceRuntimeConfig(
            image="autonomi/nos:latest-diffusers-gpu",
            name=f"{NOS_CUSTOM_SERVICE_CONTAINER_NAME}-diffusers-gpu",
            gpu=True,
            ports={
                8265: 8265,
            },
        ),
    }

    @classmethod
    def list(self, **kwargs) -> List[docker.models.containers.Container]:
        """List running docker containers."""
        containers = DockerRuntime.get().list(**kwargs)
        return [container for container in containers if container.name.startswith(NOS_CUSTOM_SERVICE_CONTAINER_NAME)]
