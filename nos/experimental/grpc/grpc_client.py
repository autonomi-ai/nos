"""
Simple gRPC client for NOS service.

Used for testing purposes and in conjunction with the NOS gRPC server (grpc_server.py).
"""
import asyncio

from google.protobuf import empty_pb2


def _nos_grpc():
    import ray

    import grpc

    ray.init(address="auto")

    from nos.experimental.grpc import import_module

    nos_service_pb2 = import_module("nos_service_pb2")
    nos_service_pb2_grpc = import_module("nos_service_pb2_grpc")

    async def send_request():
        async with grpc.aio.insecure_channel("localhost:50051") as channel:
            stub = nos_service_pb2_grpc.InferenceServiceStub(channel)

            # List models
            request = empty_pb2.Empty()
            response: nos_service_pb2.ModelListResponse = await stub.ListModels(request)
            print(response.models)

            # Create init model request
            method, model_name = "txt2vec", "openai/clip-vit-base-patch32"
            # method, model_name = "txt2img", "stabilityai/stable-diffusion-2"
            request = nos_service_pb2.InitModelRequest(model_name=model_name, min_replicas=0, max_replicas=2)

            # Issue init model request
            response: nos_service_pb2.InitModelResponse = await stub.InitModel(request)
            print(response.result)

            # Run inference
            response = await stub.Predict(
                nos_service_pb2.InferenceRequest(
                    method=method, model_name=model_name, text="a cat dancing on the grass."
                )
            )
            # Get the object ref
            result_ref = ray.cloudpickle.loads(response.result)
            # Get the object
            result = ray.get(result_ref)
            print(result)

        return response

    async def main():
        await send_request()

    asyncio.run(main())


if __name__ == "__main__":
    _nos_grpc()
