"""
Simple gRPC client for NOS service.

Used for testing purposes and in conjunction with the NOS gRPC server (grpc_server.py).
"""
from contextlib import asynccontextmanager

from nos.experimental.grpc import import_module


nos_service_pb2 = import_module("nos_service_pb2")
nos_service_pb2_grpc = import_module("nos_service_pb2_grpc")


@asynccontextmanager
async def remote_model(stub, model_name: str):
    """Remote model context manager."""
    # Create inference stub and init model
    request = nos_service_pb2.InitModelRequest(model_name=model_name, min_replicas=1, max_replicas=2)
    response: nos_service_pb2.InitModelResponse = await stub.InitModel(request)
    print(response.result)

    # Yield so that the model inference can be done
    yield

    # Delete model
    request = nos_service_pb2.DeleteModelRequest(model_name=model_name)
    response: nos_service_pb2.DeleteModelResponse = await stub.DeleteModel(request)
    print(response.result)
