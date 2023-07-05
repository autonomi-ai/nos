import docker
from nos.constants import DEFAULT_GRPC_PORT


def init(
    runtime: str = "auto", port: int = DEFAULT_GRPC_PORT, utilization: float = 0.8
) -> docker.models.containers.Container:
    """Initialize the inference server.

    Args:
        runtime (str, optional): The runtime to use (i.e. "cpu", "gpu"). Defaults to "auto".
            In "auto" mode, the runtime will be automatically detected.
        port (int, optional): The port to use for the inference server. Defaults to DEFAULT_GRPC_PORT.
        utilization (float, optional): The target cpu/memory utilization of inference server. Defaults to 0.8.
    """
    import math

    import psutil

    from nos.common.system import get_system_info, has_gpu
    from nos.logging import logger
    from nos.server.runtime import InferenceServiceRuntime

    _MIN_NUM_CPUS = 4
    _MIN_MEM_GB = 6
    _MIN_SHMEM_GB = 4

    def _check_system_requirements():
        # For now, we require at least 4 physical CPU cores and 6 GB of free memory
        sysinfo = get_system_info()
        num_cpus = sysinfo["cpu"]["cores"]["physical"]
        mem_gb = sysinfo["memory"]["available"] / 1024**3

        if num_cpus <= _MIN_NUM_CPUS:
            raise ValueError(
                f"Insufficient number of physical CPU cores ({num_cpus} cores), at least 4 cores required."
            )
        if mem_gb <= _MIN_MEM_GB:
            raise ValueError(
                f"Insufficient available system memory ({mem_gb:.1}GB), at least 6 GB of free memory required."
            )

    # Check system requirements
    _check_system_requirements()

    # Check if inference server is already running
    containers = InferenceServiceRuntime.list()
    if len(containers) > 0:
        assert (
            len(containers) == 1
        ), "Multiple inference servers running, please manually stop all containers before proceeding."
        (container,) = containers
        logger.warning(f"Inference server already running (name={container.name}, id={container.id[:12]}).")
        return container

    # Determine runtime from system
    if runtime == "auto":
        runtime = "gpu" if has_gpu() else "cpu"
    else:
        if runtime not in InferenceServiceRuntime.configs:
            raise ValueError(
                f"Invalid inference service runtime: {runtime}, available: {list(InferenceServiceRuntime.configs.keys())}"
            )

    # Start inference server
    runtime = InferenceServiceRuntime(runtime=runtime)
    logger.info(f"Starting inference service: [name={runtime.cfg.name}, runtime={runtime}]")

    # Determine number of cpus, system memory before starting container
    num_cpus = max(_MIN_NUM_CPUS, utilization * psutil.cpu_count(logical=False))
    mem_limit = max(_MIN_MEM_GB, utilization * math.floor(psutil.virtual_memory().available / 1024**3))
    logger.debug(f"Starting inference container: [num_cpus={num_cpus}, mem_limit={mem_limit}g]")

    # Start container
    container = runtime.start(
        nano_cpus=int(num_cpus * 1e9),
        mem_limit=f"{mem_limit}g",
        shm_size=f"{_MIN_SHMEM_GB}g",
        ports={f"{DEFAULT_GRPC_PORT}/tcp": port},
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
