"""Docker utilities to run containerized inference workloads
(compile/infer) in detached mode.
"""
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Union

import docker
import docker.errors
import docker.models.containers
import docker.models.images
import docker.models.volumes
import docker.types
from nos.exceptions import NosServerException
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

    @classmethod
    def list(cls, **kwargs) -> Iterable[docker.models.containers.Container]:
        """List docker containers."""
        return cls.get()._client.containers.list(**kwargs)

    def start(
        self,
        image: str,
        command: Optional[Union[str, List[str]]] = None,
        name: str = None,
        **kwargs: Any,
    ) -> docker.models.containers.Container:
        """Start docker container.

        Args:
            **kwargs: See https://docker-py.readthedocs.io/en/stable/containers.html#docker.models.containers.ContainerCollection.run
            ports (Optional[Dict[int, int]], optional): Port mapping. Defaults to None.
            environment (Optional[Dict[str, str]], optional): Environment variables. Defaults to None.
            volumes (Optional[Dict[str, str]], optional): Volume mapping. Defaults to None.
            shm_size (Optional[int], optional): Shared memory size. Defaults to None.
            detach (bool, optional): Whether to run the container in detached mode. Defaults to True.
            remove (bool, optional): Whether to remove the container when it exits. Defaults to True.
            gpu (bool, optional): Whether to start the container with GPU support. Defaults to False.

        Note (Non-standard arguments):
            gpu (bool): Whether to start the container with GPU support.

        """
        # Check if container is already running, raise error if it is
        if name and self.get_container(name) is not None:
            container = self.get_container(name)
            if container.status == "running":
                raise RuntimeError(f"Container with same name already running (name={name}).")
            else:
                logger.warning(f"Container with same name already exists, removing it (name={name}).")
                self.stop(name)

        # Validate kwargs before passing to `containers.run(...)`
        if "device_requests" in kwargs:
            raise ValueError("Use `gpu=True` instead of `device_requests`.")

        # Handle device requests (gpu=True)
        device_requests = []
        if kwargs.pop("gpu", False):
            device_requests = [DeviceRequest.get("gpu")]

        # Try starting the container, if it fails, remove it and try again
        logger.debug(f"Starting container: {name}")
        logger.debug(f"\timage: {image}")
        logger.debug(f"\tcommand: {command}")
        logger.debug(f"\tname: {name}")
        logger.debug(f"\tdevice: {device_requests}")
        for k, v in kwargs.items():
            logger.debug(f"\t{k}: {v}")

        # Start container (pass through kwargs)
        try:
            container = self._client.containers.run(
                image,
                command=command,
                name=name,
                device_requests=device_requests,
                **kwargs,
            )
            logger.debug(f"Started container [name={name}, image={container.image}, id={container.id[:12]}]")
            logger.debug(f"Get logs using `docker logs -f {container.id[:12]}`")
        except (docker.errors.APIError, docker.errors.DockerException) as exc:
            logger.error(f"Failed to start container, cleaning up container: {exc}")
            self.stop(name)
            raise NosServerException(f"Failed to start container [image={image}]")
        return container

    def stop(self, name: str, timeout: int = 30) -> docker.models.containers.Container:
        """Stop docker container."""
        try:
            container = self.get_container(name)
            if container is None:
                logger.debug(f"Container not running: {name}, exiting early.")
                return
            logger.debug(f"Removing container: [name={name}, image={container.image}, id={container.id[:12]}]")
            container.remove(force=True)
            logger.debug(f"Removed container: [name={name}, image={container.image}, id={container.id[:12]}]")
        except (docker.errors.APIError, docker.errors.DockerException) as exc:
            logger.error(f"Failed to stop container: {exc}")
        return container

    def get_container_id(self, name: str) -> Optional[str]:
        """Get the runtime container ID."""
        container = self.get_container(name)
        return container.id if container else None

    def get_container(self, id_or_name: str) -> docker.models.containers.Container:
        """Get container by id or name."""
        try:
            return self._client.containers.get(id_or_name)
        except docker.errors.NotFound:
            return None

    def get_container_status(self, id_or_name: str) -> Optional[str]:
        """Get container status by id or name."""
        container = self.get_container(id_or_name)
        return container.status if container else None

    def get_container_logs(self, name: str, **kwargs) -> Iterable[str]:
        """Get container logs."""
        try:
            container = self.get_container(name)
            if container is None:
                return iter([])

            for line in container.logs(stream=True):
                yield line.decode("utf-8")
        except (docker.errors.APIError, docker.errors.DockerException) as exc:
            logger.error(f"Failed to get container logs: {exc}")
            raise NosServerException("Failed to get container logs")
