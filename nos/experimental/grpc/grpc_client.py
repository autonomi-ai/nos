"""
Simple gRPC client for NOS service.

Used for testing purposes and in conjunction with the NOS gRPC server (grpc_server.py).
"""
import asyncio
from contextlib import asynccontextmanager

import numpy as np
import ray
from google.protobuf import empty_pb2
from PIL import Image
from tqdm import tqdm

import grpc
from nos.experimental.grpc import import_module
from nos.test.utils import NOS_TEST_IMAGE


nos_service_pb2 = import_module("nos_service_pb2")
nos_service_pb2_grpc = import_module("nos_service_pb2_grpc")


@asynccontextmanager
async def remote_model(model_name: str, address: str = "localhost:50051"):
    """Remote model context manager."""
    async with grpc.aio.insecure_channel(address) as channel:
        # Create inference stub and init model
        stub = nos_service_pb2_grpc.InferenceServiceStub(channel)
        request = nos_service_pb2.InitModelRequest(model_name=model_name, min_replicas=1, max_replicas=2)
        response: nos_service_pb2.InitModelResponse = await stub.InitModel(request)
        print(response.result)

        # Yield the gprc stub once the model is initialized
        yield stub

        # Delete model
        request = nos_service_pb2.DeleteModelRequest(model_name=model_name)
        response: nos_service_pb2.DeleteModelResponse = await stub.DeleteModel(request)
        print(response.result)


def _nos_grpc():
    img = Image.open(NOS_TEST_IMAGE)
    img = np.array(img)

    async def send_request():
        async with grpc.aio.insecure_channel("localhost:50051") as channel:
            stub = nos_service_pb2_grpc.InferenceServiceStub(channel)

            # List models
            request = empty_pb2.Empty()
            response: nos_service_pb2.ModelListResponse = await stub.ListModels(request)
            print(response.models)

        # TXT2VEC
        method, model_name = "txt2vec", "openai/clip-vit-base-patch32"
        async with remote_model(model_name=model_name) as stub:
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
        async with remote_model(model_name=model_name) as stub:
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
        # Create init model request
        method, model_name = "txt2img", "stabilityai/stable-diffusion-2"
        async with remote_model(model_name=model_name) as stub:
            for _ in tqdm(range(10), desc=f"Bench [method{method}, model_name={model_name}]"):
                response = await stub.Predict(
                    nos_service_pb2.InferenceRequest(
                        method=method,
                        model_name=model_name,
                        text_request=nos_service_pb2.TextRequest(text="a cat dancing on the grass."),
                    )
                )
                ray.cloudpickle.loads(response.result)

    async def main():
        await send_request()

    asyncio.run(main())


if __name__ == "__main__":
    _nos_grpc()
