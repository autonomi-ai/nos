"""Fully integrated gRPC client test with gRPC-based inference server.

The test spins up a gRPC inferennce server and then sends requests to it using the gRPC client.

"""
import grpc
import numpy as np
import pytest
import ray
from google.protobuf import empty_pb2
from PIL import Image
from tqdm import tqdm

from nos.experimental.grpc import DEFAULT_GRPC_PORT, InferenceService, import_module, remote_model
from nos.test.utils import NOS_TEST_IMAGE


GRPC_TEST_PORT = DEFAULT_GRPC_PORT + 1

nos_service_pb2 = import_module("nos_service_pb2")
nos_service_pb2_grpc = import_module("nos_service_pb2_grpc")


def grpc_server(port: int = GRPC_TEST_PORT):
    from loguru import logger

    logger.info(f"Starting gRPC test server on port: {port}")
    server = grpc.aio.server()
    nos_service_pb2_grpc.add_InferenceServiceServicer_to_server(InferenceService(), server)
    server.add_insecure_port(f"[::]:{port}")
    return server


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_e2e_grpc_client_and_server():
    img = Image.open(NOS_TEST_IMAGE)
    img = np.array(img)

    async def send_request(stub):
        # List models
        request = empty_pb2.Empty()
        response: nos_service_pb2.ModelListResponse = await stub.ListModels(request)
        print(response.models)

        # TXT2VEC
        method, model_name = "txt2vec", "openai/clip-vit-base-patch32"
        async with remote_model(stub, model_name=model_name):
            for _ in tqdm(range(10), desc=f"Bench [method{method}, model_name={model_name}]"):
                response = await stub.Predict(
                    nos_service_pb2.InferenceRequest(
                        method=method,
                        model_name=model_name,
                        text_request=nos_service_pb2.TextRequest(text="a cat dancing on the grass."),
                    )
                )
                ray.cloudpickle.loads(response.result)

        # IMG2VEC
        method, model_name = "img2vec", "openai/clip-vit-base-patch32"
        async with remote_model(stub, model_name=model_name):
            for _ in tqdm(range(10), desc=f"Bench [method{method}, model_name={model_name}]"):
                response = await stub.Predict(
                    nos_service_pb2.InferenceRequest(
                        method=method,
                        model_name=model_name,
                        image_request=nos_service_pb2.ImageRequest(image_bytes=ray.cloudpickle.dumps(img)),
                    )
                )
                ray.cloudpickle.loads(response.result)

        # TXT2IMG
        method, model_name = "txt2img", "stabilityai/stable-diffusion-2"
        async with remote_model(stub, model_name=model_name):
            for _ in tqdm(range(10), desc=f"Bench [method{method}, model_name={model_name}]"):
                response = await stub.Predict(
                    nos_service_pb2.InferenceRequest(
                        method=method,
                        model_name=model_name,
                        text_request=nos_service_pb2.TextRequest(text="a cat dancing on the grass."),
                    )
                )
                ray.cloudpickle.loads(response.result)

    # Start the gRPC server
    server = grpc_server(port=GRPC_TEST_PORT)
    async with grpc.aio.insecure_channel(f"localhost:{GRPC_TEST_PORT}") as channel:
        stub = nos_service_pb2_grpc.InferenceServiceStub(channel)
        await server.start()
        await send_request(stub)
        await server.stop(grace=None)
