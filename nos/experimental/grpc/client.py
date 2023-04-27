"""
Simple gRPC client for NOS service.

Used for testing purposes and in conjunction with the NOS gRPC server (grpc_server.py).
"""
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from nos.constants import DEFAULT_GRPC_PORT  # noqa F401
from nos.executors.docker import DockerExecutor
from nos.experimental.grpc import import_module
from nos.logging import LOGGING_LEVEL, logger


nos_service_pb2 = import_module("nos_service_pb2")
nos_service_pb2_grpc = import_module("nos_service_pb2_grpc")


NOS_DOCKER_IMAGE_GPU = "autonomi-ai/nos:latest-gpu"
NOS_DOCKER_IMAGE_CPU = "autonomi-ai/nos:latest-cpu"

NOS_GRPC_SERVER_CONTAINER_NAME = "nos-grpc-server"
NOS_GRPC_SERVER_CMD = "nos-grpc-server"


@asynccontextmanager
async def remote_model(stub, model_name: str):
    """Remote model context manager."""
    # Create inference stub and init model
    request = nos_service_pb2.InitModelRequest(model_name=model_name, min_replicas=1, max_replicas=2)
    response: nos_service_pb2.InitModelResponse = await stub.InitModel(request)
    logger.info(f"Init Model response: {response}")

    # Yield so that the model inference can be done
    yield

    # Delete model
    request = nos_service_pb2.DeleteModelRequest(model_name=model_name)
    response: nos_service_pb2.DeleteModelResponse = await stub.DeleteModel(request)
    logger.info(f"Delete Model response: {response}")


@dataclass
class InferenceRuntime:
    """Inference docker runtime."""

    _executor: DockerExecutor = None

    def __init__(self):
        """Initialize InferenceRuntime."""
        self._executor = DockerExecutor.get()
        logger.info(f"Initialized InferenceRuntime: {self._executor}")

    def start(self, detach: bool = True, gpu: bool = False):
        """Start the inference client.

        Args:
            gpu (bool, optional): Whether to start the client with GPU support. Defaults to False.
        """
        image = NOS_DOCKER_IMAGE_GPU if gpu else NOS_DOCKER_IMAGE_CPU
        logger.info(f"Starting inference client with image: {image}")
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
            shm_size="4g",
            detach=detach,
            remove=True,
            gpu=gpu,
        )
        logger.info(f"Started inference client: {self._executor}")

    def stop(self):
        """Stop the inference client."""
        self._executor.stop(NOS_GRPC_SERVER_CONTAINER_NAME)
        logger.info(f"Stopped inference client: {self._executor}")

    def get_logs(self):
        """Get the inference client logs."""
        return self._executor.get_logs(NOS_GRPC_SERVER_CONTAINER_NAME)
