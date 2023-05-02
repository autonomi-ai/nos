"""
Simple gRPC client for NOS service.

Used for testing purposes and in conjunction with the NOS gRPC server (grpc_server.py).
"""
from contextlib import contextmanager

from nos.experimental.grpc import import_module
from nos.logging import logger


nos_service_pb2 = import_module("nos_service_pb2")
nos_service_pb2_grpc = import_module("nos_service_pb2_grpc")


@contextmanager
def InferenceSession(stub, model_name: str, num_replicas: int = 1):
    """Remote model context manager."""
    # Create inference stub and init model
    request = nos_service_pb2.InitModelRequest(model_name=model_name, num_replicas=num_replicas)
    response: nos_service_pb2.InitModelResponse = stub.InitModel(request)
    logger.info(f"Init Model response: {response}")
    # Yield so that the model inference can be done
    yield
    # Delete model
    request = nos_service_pb2.DeleteModelRequest(model_name=model_name)
    response: nos_service_pb2.DeleteModelResponse = stub.DeleteModel(request)
    logger.info(f"Delete Model response: {response}")
