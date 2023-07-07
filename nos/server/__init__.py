import logging
import math
from typing import Optional

import psutil

import docker
from nos.constants import DEFAULT_GRPC_PORT


def init(
    runtime: str = "auto",
    port: int = DEFAULT_GRPC_PORT,
    utilization: float = 0.8,
    pull: bool = True,
    logging_level: int = logging.INFO,
    tag: Optional[str] = None,
) -> docker.models.containers.Container:
    """Initialize the inference server.

    Args:
        runtime (str, optional): The runtime to use (i.e. "cpu", "gpu"). Defaults to "auto".
            In "auto" mode, the runtime will be automatically detected.
        port (int, optional): The port to use for the inference server. Defaults to DEFAULT_GRPC_PORT.
        utilization (float, optional): The target cpu/memory utilization of inference server. Defaults to 0.8.
        pull (bool, optional): Pull the docker image before starting the inference server. Defaults to True.
        logging_level (int, optional): The logging level to use for the inference server. Defaults to logging.INFO.
        tag (str, optional): The tag of the docker image to use ("latest"). Defaults to None, where the
            appropriate version is used.
    """
    from nos.common.system import has_docker, has_gpu
    from nos.logging import logger
    from nos.server.runtime import DockerRuntime, InferenceServiceRuntime
    from nos.version import __version__

    # Check arguments
    available_runtimes = list(InferenceServiceRuntime.configs.keys()) + ["auto"]
    if runtime not in available_runtimes:
        raise ValueError(f"Invalid inference service runtime: {runtime}, available: {available_runtimes}")

    if utilization <= 0.25 or utilization > 1:
        raise ValueError(f"Invalid utilization: {utilization}, must be in (0.25, 1].")

    if logging_level not in (logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL):
        raise ValueError(f"Invalid logging level: {logging_level}")

    if tag is None:
        tag = __version__
    else:
        if not isinstance(tag, str):
            raise ValueError(f"Invalid tag: {tag}, must be a string.")
        raise NotImplementedError("Custom tags are not yet supported.")

    _MIN_NUM_CPUS = 4
    _MIN_MEM_GB = 6
    _MIN_SHMEM_GB = 4

    def _check_system_requirements():
        if not has_docker():
            raise RuntimeError("Docker not found, please install docker before proceeding.")

        # For now, we require at least 4 physical CPU cores and 6 GB of free memory
        cl = DockerRuntime.get()._client
        num_cpus = cl.info().get("NCPU", psutil.cpu_count(logical=False))
        mem_avail = (
            min(cl.info().get("MemTotal", psutil.virtual_memory().total), psutil.virtual_memory().available)
            / 1024**3
        )
        logger.debug(f"Checking system requirements: [num_cpus={num_cpus}, mem_avail={mem_avail:.1f}GB]")
        if num_cpus < _MIN_NUM_CPUS:
            raise ValueError(
                f"Insufficient number of physical CPU cores ({num_cpus} cores), at least 4 cores required."
            )
        if mem_avail < _MIN_MEM_GB:
            raise ValueError(
                f"Insufficient available system memory ({mem_avail:.1f}GB), at least 6 GB of free memory required."
            )

    # Check system requirements
    _check_system_requirements()

    # Determine runtime from system
    if runtime == "auto":
        runtime = "gpu" if has_gpu() else "cpu"

    # Check if inference server is already running
    containers = InferenceServiceRuntime.list()
    if len(containers) > 0:
        assert (
            len(containers) == 1
        ), "Multiple inference servers running, please manually stop all containers before proceeding."
        (container,) = containers
        logger.warning(f"Inference server already running (name={container.name}, id={container.id[:12]}).")
        return container

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
    container = runtime.start(
        nano_cpus=int(num_cpus * 1e9),
        mem_limit=f"{mem_limit}g",
        shm_size=f"{_MIN_SHMEM_GB}g",
        ports={f"{DEFAULT_GRPC_PORT}/tcp": port},
        environment={
            "NOS_LOGGING_LEVEL": logging.getLevelName(logging_level),
        },
    )
    logger.info(f"Inference service started: [name={runtime.cfg.name}, runtime={runtime}, id={container.id[:12]}]")
    return container


def shutdown() -> docker.models.containers.Container:
    """Shutdown the inference server."""
    from nos.logging import logger
    from nos.server.runtime import InferenceServiceRuntime

    # Check if inference server is already running
    containers = InferenceServiceRuntime.list()
    if not len(containers):
        raise RuntimeError("Inference server not running, nothing to shutdown.")
    if len(containers) > 1:
        raise RuntimeError("Multiple inference servers running, please manually stop all containers.")
    # Shutdown inference server
    (container,) = containers
    logger.info(f"Stopping inference service: [name={container.name}, id={container.id[:12]}]")
    try:
        container.remove(force=True)
    except Exception as e:
        raise RuntimeError(f"Failed to shutdown inference server: {e}")
    logger.info(f"Inference service stopped: [name={container.name}, id={container.id[:12]}]")
    return container


def _pull_image(image: str, quiet: bool = False, platform: str = None) -> str:
    """Pull the latest inference server image."""
    import subprocess
    from collections import deque

    import rich.status

    import docker.errors
    from nos.logging import logger
    from nos.server.runtime import DockerRuntime

    try:
        DockerRuntime.get()._client.images.get(image)
        logger.info(f"Found up-to-date server image: {image}")
    except docker.errors.ImageNotFound:
        try:
            logger.info(f"Pulling new server image: {image} (this may take a while).")
            # use subprocess to pull image and stream output
            proc = subprocess.Popen(
                f"docker pull {image}", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True
            )
            status_q = deque(maxlen=25)
            status_str = f"Pulling new server image: {image} (this may take a while)."
            with rich.status.Status("[bold white]" + status_str + "[/bold white]") as status:
                while proc.stdout.readable():
                    line = proc.stdout.readline()
                    if not line:
                        break
                    status_q.append(line.decode("utf-8").strip())
                    status.update("[bold white]" + f"{status_str}\n\t" + "\n\t".join(status_q) + "[/bold white]")
            proc.wait()
            logger.info(f"Pulled new server image: {image}")
        except Exception as exc:
            logger.error(f"Failed to pull image: {image}, exiting early: {exc}")
            raise Exception(f"Failed to pull image: {image}, please mnaully pull image via `docker pull {image}`")
