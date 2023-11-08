import logging
import math
import platform
import subprocess
from collections import deque
from typing import List, Optional, Union

import psutil
import rich.status

import docker
import docker.errors
import docker.models.containers
from nos.common.shm import NOS_SHM_ENABLED
from nos.constants import DEFAULT_GRPC_PORT
from nos.logging import logger
from nos.version import __version__

from ._docker import DockerRuntime
from ._runtime import InferenceServiceRuntime


_MIN_NUM_CPUS = 4
_MIN_MEM_GB = 6
_MIN_SHMEM_GB = 4


def _check_system_requirements(runtime: str):
    """Check system requirements."""
    from nos.common.system import has_docker, has_gpu, has_nvidia_docker_runtime_enabled

    logger.debug(f"Checking system requirements, [nos={__version__}].")
    if not has_docker():
        raise RuntimeError("Docker not found, please install docker before proceeding.")

    if runtime == "gpu":
        if not has_gpu():
            raise RuntimeError("GPU not found, please install CUDA drivers before proceeding.")
        if not has_nvidia_docker_runtime_enabled():
            raise RuntimeError(
                "NVIDIA Docker runtime not enabled, please enable NVIDIA Docker runtime before proceeding."
            )

    # For now, we require at least 4 physical CPU cores and 6 GB of free memory
    cl = DockerRuntime.get()._client
    num_cpus = cl.info().get("NCPU", psutil.cpu_count(logical=False))
    mem_avail = (
        min(cl.info().get("MemTotal", psutil.virtual_memory().total), psutil.virtual_memory().available) / 1024**3
    )
    logger.debug(f"Checking system requirements: [num_cpus={num_cpus}, mem_avail={mem_avail:.1f}GB]")
    if num_cpus < _MIN_NUM_CPUS:
        raise ValueError(f"Insufficient number of physical CPU cores ({num_cpus} cores), at least 4 cores required.")
    if mem_avail < _MIN_MEM_GB:
        raise ValueError(
            f"Insufficient available system memory ({mem_avail:.1f}GB), at least 6 GB of free memory required."
        )


def init(
    runtime: str = "auto",
    port: int = DEFAULT_GRPC_PORT,
    utilization: float = 1.0,
    pull: bool = True,
    logging_level: Union[int, str] = logging.INFO,
    tag: Optional[str] = None,
) -> docker.models.containers.Container:
    """Initialize the NOS inference server (as a docker daemon).

    The method first checks to see if your system requirements are met, before pulling the NOS docker image from Docker Hub
    (if necessary) and starting the inference server (as a docker daemon). You can also specify the runtime to use (i.e. "cpu", "gpu"),
    and the port to use for the inference server.


    Args:
        runtime (str, optional): The runtime to use (i.e. "auto", "local", "cpu", "gpu"). Defaults to "auto".
            In "auto" mode, the runtime will be automatically detected.
        port (int, optional): The port to use for the inference server. Defaults to DEFAULT_GRPC_PORT.
        utilization (float, optional): The target cpu/memory utilization of inference server. Defaults to 1.
        pull (bool, optional): Pull the docker image before starting the inference server. Defaults to True.
        logging_level (Union[int, str], optional): The logging level to use. Defaults to logging.INFO.
            Optionally, a string can be passed (i.e. "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL").
        tag (str, optional): The tag of the docker image to use ("latest"). Defaults to None, where the
            appropriate version is used.
    """
    # Check arguments
    available_runtimes = list(InferenceServiceRuntime.configs.keys()) + ["auto", "local"]
    if runtime not in available_runtimes:
        raise ValueError(f"Invalid inference service runtime: {runtime}, available: {available_runtimes}")

    # If runtime is "local", return early with ray executor
    if runtime == "local":
        from nos.executors.ray import RayExecutor

        executor = RayExecutor.get()
        executor.init()
        return

    # Check arguments
    if utilization <= 0.25 or utilization > 1:
        raise ValueError(f"Invalid utilization: {utilization}, must be in (0.25, 1].")

    if not isinstance(logging_level, (int, str)):
        raise ValueError(f"Invalid logging level: {logging_level}, must be an integer or string.")
    if isinstance(logging_level, int):
        logging_level = logging.getLevelName(logging_level)
    if logging_level not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
        raise ValueError(f"Invalid logging level: {logging_level}")

    if tag is None:
        tag = __version__
    else:
        if not isinstance(tag, str):
            raise ValueError(f"Invalid tag: {tag}, must be a string.")
        raise NotImplementedError("Custom tags are not yet supported.")

    # Determine runtime from system
    if runtime == "auto":
        runtime = InferenceServiceRuntime.detect()
        logger.debug(f"Auto-detected system runtime: {runtime}")
    else:
        if runtime not in InferenceServiceRuntime.configs:
            raise ValueError(
                f"Invalid inference service runtime: {runtime}, available: {list(InferenceServiceRuntime.configs.keys())}"
            )

    # Check if the latest inference server is already running
    # If the running container's tag is inconsistent with the current version,
    # we will shutdown the running container and start a new one.
    containers = InferenceServiceRuntime.list()
    if len(containers) == 1:
        logger.debug("Found an existing inference server running, checking if it is the latest version.")
        if InferenceServiceRuntime.configs[runtime].image not in containers[0].image.tags:
            logger.info(
                "Active inference server is not the latest version, shutting down before starting the latest one."
            )
            _stop_container(containers[0])
        else:
            (container,) = containers
            logger.info(
                f"Inference server already running (name={container.name}, image={container.image}, id={container.id[:12]})."
            )
            return container
    elif len(containers) > 1:
        logger.warning("""Multiple inference servers running, please report this issue to the NOS maintainers.""")
        for container in containers:
            _stop_container(container)
    else:
        logger.debug("No existing inference server found, starting a new one.")

    # Check system requirements
    # Note: we do this after checking if the latest
    # inference server is already running for convenience.
    _check_system_requirements(runtime)

    # Pull docker image (if necessary)
    if pull:
        _pull_image(InferenceServiceRuntime.configs[runtime].image)

    # Start inference server
    runtime = InferenceServiceRuntime(runtime=runtime)
    logger.info(f"Starting inference service: [name={runtime.cfg.name}, runtime={runtime}]")

    # Determine number of cpus, system memory before starting container
    # Note (spillai): MacOSX compatibility issue where docker does not have access to
    # the correct number of physical cores and memory.
    cl = DockerRuntime.get()._client
    num_cpus = cl.info().get("NCPU", psutil.cpu_count(logical=False))
    num_cpus = max(_MIN_NUM_CPUS, utilization * num_cpus)
    mem_limit = (
        min(cl.info().get("MemTotal", psutil.virtual_memory().total), psutil.virtual_memory().available) / 1024**3
    )
    mem_limit = max(_MIN_MEM_GB, utilization * math.floor(mem_limit))
    logger.debug(f"Starting inference container: [num_cpus={num_cpus}, mem_limit={mem_limit}g]")

    # Start container
    # TOFIX (spillai): If macosx, shared memory is not supported
    shm_enabled = NOS_SHM_ENABLED if platform.system() == "Linux" else False
    container = runtime.start(
        nano_cpus=int(num_cpus * 1e9),
        mem_limit=f"{mem_limit}g",
        shm_size=f"{_MIN_SHMEM_GB}g",
        ports={f"{DEFAULT_GRPC_PORT}/tcp": port},
        environment={
            "NOS_LOGGING_LEVEL": logging_level,
            "NOS_SHM_ENABLED": int(shm_enabled),
        },
    )
    logger.info(
        f"Inference service started: [name={runtime.cfg.name}, runtime={runtime}, image={container.image}, id={container.id[:12]}]"
    )
    return container


def shutdown() -> Optional[Union[docker.models.containers.Container, List[docker.models.containers.Container]]]:
    """Shutdown the inference server."""
    # Check if inference server is already running
    containers = InferenceServiceRuntime.list()
    if len(containers) == 1:
        (container,) = containers
        _stop_container(container)
        return container
    if len(containers) > 1:
        logger.warning("""Multiple inference servers running, please report this issue to the NOS maintainers.""")
        for container in containers:
            _stop_container(container)
        return containers
    logger.info("No active inference servers found, ignoring shutdown.")
    return None


def _stop_container(container: docker.models.containers.Container) -> None:
    """Force stop containers."""
    logger.info(
        f"Stopping inference service: [name={container.name}, image={container.image}, id={container.id[:12]}]"
    )
    try:
        container.remove(force=True)
    except Exception as e:
        raise RuntimeError(f"Failed to shutdown inference server: {e}")
    logger.info(f"Inference service stopped: [name={container.name}, image={container.image}, id={container.id[:12]}]")


def _pull_image(image: str, quiet: bool = False, platform: str = None) -> str:
    """Pull the latest inference server image."""
    try:
        DockerRuntime.get()._client.images.get(image)
        logger.info(f"Found up-to-date server image: {image}")
    except docker.errors.ImageNotFound:
        try:
            logger.info(f"Pulling new server image: {image} (this may take a while).")
            proc = subprocess.run(f"docker pull {image}", shell=True)
            if proc.returncode != 0:
                logger.exception(f"Failed to pull docker image, e={proc.stderr}")
                raise RuntimeError(f"Failed to pull docker image, e={proc.stderr}")
            logger.info(f"Pulled new server image: {image}")
        except (docker.errors.APIError, docker.errors.DockerException) as exc:
            logger.error(f"Failed to pull image: {image}, exiting early: {exc}")
            raise Exception(f"Failed to pull image: {image}, please mnaully pull image via `docker pull {image}`")
