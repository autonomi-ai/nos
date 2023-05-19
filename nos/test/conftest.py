from concurrent import futures

import grpc
import pytest

from nos.client import DEFAULT_GRPC_PORT, InferenceClient
from nos.executors.ray import RayExecutor
from nos.protoc import import_module
from nos.server import InferenceService


GRPC_TEST_PORT = DEFAULT_GRPC_PORT + 1

nos_service_pb2 = import_module("nos_service_pb2")
nos_service_pb2_grpc = import_module("nos_service_pb2_grpc")


@pytest.fixture(scope="session")
def ray_executor():
    executor = RayExecutor.get()
    executor.init()

    yield executor

    executor.stop()


@pytest.fixture(scope="session")
def test_grpc_server():
    from loguru import logger

    logger.info(f"Starting gRPC test server on port: {GRPC_TEST_PORT}")
    options = [
        ("grpc.max_message_length", 512 * 1024 * 1024),
        ("grpc.max_send_message_length", 512 * 1024 * 1024),
        ("grpc.max_receive_message_length", 512 * 1024 * 1024),
    ]
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1), options=options)
    nos_service_pb2_grpc.add_InferenceServiceServicer_to_server(InferenceService(), server)
    server.add_insecure_port(f"[::]:{GRPC_TEST_PORT}")
    server.start()
    yield server
    server.stop(grace=None)


@pytest.fixture(scope="function")
def test_grpc_client():
    yield InferenceClient(f"localhost:{GRPC_TEST_PORT}")
