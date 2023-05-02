"""Fully integrated gRPC client test with gRPC-based inference server.

The test spins up a gRPC inferennce server and then sends requests to it using the gRPC client.

"""
from concurrent import futures

import grpc
import numpy as np
import pytest
import ray
from google.protobuf import empty_pb2
from PIL import Image
from tqdm import tqdm

from nos.experimental.grpc import DEFAULT_GRPC_PORT, InferenceService, InferenceSession, import_module
from nos.test.utils import NOS_TEST_IMAGE


GRPC_TEST_PORT = DEFAULT_GRPC_PORT + 1

nos_service_pb2 = import_module("nos_service_pb2")
nos_service_pb2_grpc = import_module("nos_service_pb2_grpc")


def grpc_server(port: int = GRPC_TEST_PORT):
    from loguru import logger

    logger.info(f"Starting gRPC test server on port: {port}")
    options = [
        ("grpc.max_message_length", 512 * 1024 * 1024),
        ("grpc.max_send_message_length", 512 * 1024 * 1024),
        ("grpc.max_receive_message_length", 512 * 1024 * 1024),
    ]
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1), options=options)
    nos_service_pb2_grpc.add_InferenceServiceServicer_to_server(InferenceService(), server)
    server.add_insecure_port(f"[::]:{port}")
    return server


@pytest.mark.e2e
def test_e2e_grpc_client_and_server():
    img = Image.open(NOS_TEST_IMAGE)
    img = np.array(img)

    def send_request(stub):
        # List models
        request = empty_pb2.Empty()
        response: nos_service_pb2.ModelListResponse = stub.ListModels(request)
        print(response.models)

        # TXT2VEC
        method, model_name = "txt2vec", "openai/clip-vit-base-patch32"
        with InferenceSession(stub, model_name=model_name):
            for _ in tqdm(range(10), desc=f"Bench [method{method}, model_name={model_name}]"):
                response = stub.Predict(
                    nos_service_pb2.InferenceRequest(
                        method=method,
                        model_name=model_name,
                        text_request=nos_service_pb2.TextRequest(text="a cat dancing on the grass."),
                    )
                )
                ray.cloudpickle.loads(response.result)

        # IMG2VEC
        method, model_name = "img2vec", "openai/clip-vit-base-patch32"
        with InferenceSession(stub, model_name=model_name):
            for _ in tqdm(range(10), desc=f"Bench [method{method}, model_name={model_name}]"):
                response = stub.Predict(
                    nos_service_pb2.InferenceRequest(
                        method=method,
                        model_name=model_name,
                        image_request=nos_service_pb2.ImageRequest(image_bytes=ray.cloudpickle.dumps(img)),
                    )
                )
                ray.cloudpickle.loads(response.result)

        # TXT2IMG
        method, model_name = "txt2img", "stabilityai/stable-diffusion-2"
        with InferenceSession(stub, model_name=model_name):
            for _ in tqdm(range(10), desc=f"Bench [method{method}, model_name={model_name}]"):
                response = stub.Predict(
                    nos_service_pb2.InferenceRequest(
                        method=method,
                        model_name=model_name,
                        text_request=nos_service_pb2.TextRequest(text="a cat dancing on the grass."),
                    )
                )
                ray.cloudpickle.loads(response.result)

    # Start the gRPC server and send requests to it
    server = grpc_server(port=GRPC_TEST_PORT)
    with grpc.insecure_channel(f"localhost:{GRPC_TEST_PORT}") as channel:
        stub = nos_service_pb2_grpc.InferenceServiceStub(channel)
        server.start()
        send_request(stub)
        server.stop(grace=None)
