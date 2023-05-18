"""Docker utilities to run containerized inference workloads
(compile/infer) in detached mode.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import docker
import docker.errors
import docker.models.containers
import docker.models.images
import docker.models.volumes
import docker.types
from nos.logging import logger


@dataclass
class DockerDeviceRequest:
    """Docker device request."""

    device_ids: List[List[str]]
    capabilities: List[List[str]]


@dataclass
class DeviceRequest:
    """Device request."""

    configs = {
        "gpu": docker.types.DeviceRequest(
            device_ids=["all"],
            capabilities=[["gpu"]],
        ),
    }

    @classmethod
    def get(cls, device: str) -> "DockerDeviceRequest":
        """Get device request."""
        try:
            return cls.configs[device]
        except KeyError:
            raise ValueError(f"Invalid DockerDeviceRequest: {device}")


@dataclass
class DockerRuntime:
    """
    Docker runtime for running containerized inference workloads.
    """

    _instance: "DockerRuntime" = None
    _client: docker.DockerClient = None

    def __init__(self):
        """Initialize DockerExecutor."""
        self._client = docker.from_env()

    @classmethod
    def get(cls: "DockerRuntime") -> "DockerRuntime":
        """Get DockerRuntime instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def start(
        self,
        image: str,
        container_name: str,
        ports: Optional[Dict[int, int]] = None,
        command: Optional[List[str]] = None,
        environment: Optional[Dict[str, str]] = None,
        volumes: Optional[Dict[str, str]] = None,
        detach: bool = True,
        shm_size: str = "4g",
        **kwargs: Any,
    ) -> docker.models.containers.Container:
        """Start docker container."""
        container = self.get_container(container_name)

        # If container is already running, return it
        if container is not None:
            if container.status == "running":
                logger.info(f"Container already running: {container_name} id={container.id[:12]}")
                logger.info(f"Get logs using `docker logs -f {container.id[:12]}`")
                return container
            else:
                logger.info(f"Removing existing container: {container_name}")
                container.remove(force=True)

        # If container is not running, start it in detached mode
        device_requests = []
        if kwargs.pop("gpu", False):
            device_requests = [DeviceRequest.get("gpu")]

        # Try starting the container, if it fails, remove it and try again
        logger.info(f"Starting container: {container_name}")
        logger.debug(f"\timage: {image}")
        logger.debug(f"\tports: {ports}")
        logger.debug(f"\tcommand: {command}")
        logger.debug(f"\tenvironment: {environment}")
        logger.debug(f"\tvolumes: {volumes}")
        logger.debug(f"\tshm_size: {shm_size}")
        logger.debug(f"\tdevice: {device_requests}")
        try:
            container = self._client.containers.run(
                image,
                ports=ports,
                command=command,
                detach=detach,
                name=container_name,
                volumes=volumes,
                device_requests=device_requests,
                environment=environment,
                shm_size=shm_size,
            )
        except (docker.errors.APIError, docker.errors.DockerException) as exc:
            if container is not None:
                container.remove(force=True)
            logger.error(f"Failed to start container: {exc}")
            raise exc
        logger.info(f"Started container: {container_name}")
        logger.info(f"Get logs using `docker logs -f {container.id[:12]}`")
        return container

    def stop(self, container_name: str, timeout: int = 30) -> docker.models.containers.Container:
        """Stop docker container."""
        try:
            container = self.get_container(container_name)
            if container is None:
                logger.info(f"Container not running: {container_name}, exiting early.")
                return
            logger.info(f"Stopping container: {container_name}")
            container.stop(timeout=timeout)
            logger.info(f"Stopped container: {container_name}")
            container.remove(force=True)
            logger.info(f"Removed container: {container_name}")
        except (docker.errors.APIError, docker.errors.DockerException) as exc:
            logger.error(f"Failed to stop container: {exc}")
            raise exc
        return container

    def get_container(self, container_name: str) -> docker.models.containers.Container:
        """Get container by name."""
        try:
            return self._client.containers.get(container_name)
        except docker.errors.NotFound:
            return None

    def get_container_status(self, container_name: str) -> str:
        """Get container status by name."""
        container = self.get_container(container_name)
        if container is None:
            return None
        return container.status

    def get_logs(self, container_name: str, tail: int = 10, **kwargs: Any) -> str:
        """Get container logs."""
        try:
            container = self.get_container(container_name)
            if container is None:
                return ""

            return container.logs(tail=tail).decode("utf-8")
        except (docker.errors.APIError, docker.errors.DockerException) as exc:
            logger.error(f"Failed to get container logs: {exc}")
            raise exc

    def get_ports(self, container_name: str) -> Dict[int, int]:
        """Get container ports."""
        try:
            container = self.get_container(container_name)
            if container is None:
                return {}
            return {container_p: host_p.get("HostPort") for container_p, host_p in container.ports.items()}
        except (docker.errors.APIError, docker.errors.DockerException) as exc:
            logger.error(f"Failed to get container ports: {exc}")
            raise exc
