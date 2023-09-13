from typing import List

import docker
from nos.server._docker import DockerRuntime
from nos.server._runtime import InferenceServiceRuntime, InferenceServiceRuntimeConfig


NOS_CUSTOM_SERVICE_CONTAINER_NAME = "nos-custom-service"


class CustomServiceRuntimeConfig(InferenceServiceRuntimeConfig):
    pass


class CustomServiceRuntime(InferenceServiceRuntime):
    """Custom service runtime, inherits from InferenceServiceRuntime."""

    configs = {
        "mmdet-gpu": CustomServiceRuntimeConfig(
            image="autonomi/nos:latest-mmdet-gpu",
            name=f"{NOS_CUSTOM_SERVICE_CONTAINER_NAME}-mmdet-gpu",
            gpu=True,
            kwargs={
                "nano_cpus": int(8e9),
                "mem_limit": "12g",
            },
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
