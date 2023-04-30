import numpy as np
import ray
import rich.console
import rich.status
from google.protobuf import empty_pb2

import grpc
from nos import hub
from nos.exceptions import ModelNotFoundError
from nos.experimental.grpc import import_module
from nos.hub import MethodType
from nos.logging import logger


nos_service_pb2 = import_module("nos_service_pb2")
nos_service_pb2_grpc = import_module("nos_service_pb2_grpc")


class InferenceService(nos_service_pb2_grpc.InferenceServiceServicer):
    """
    Experimental gRPC-based inference service.

    This service is used to serve models over gRPC.

    Refer to the bring-your-own-schema section:
    https://docs.ray.io/en/master/serve/direct-ingress.html?highlight=grpc#bring-your-own-schema
    """

    def __init__(self):
        self.models = hub.list()
        self.model_spec = None
        self.model_handle = None

    def init_model(self, model_name: str):
        """Initialize the model."""
        # Load the model spec
        try:
            self.model_spec = hub.load_spec(model_name)
            logger.info(f"Loaded model spec: {self.model_spec}")
        except Exception as e:
            raise ModelNotFoundError(f"Failed to load model spec: {model_name}, {e}")

        # Create the serve deployment from the model handle
        model_cls = self.model_spec.cls
        actor_options = {"num_gpus": 1}
        logger.info(f"Creating actor: {actor_options}")
        actor_cls = ray.remote(**actor_options)(model_cls)
        self.model_handle = actor_cls.remote()
        logger.info(f"Created actor: {self.model_handle}, type={type(self.model_handle)}")

    def delete_model(self, model_name: str):
        """Delete the model."""
        logger.info(f"Deleting model: {model_name}")
        ray.kill(self.model_handle)
        del self.model_handle
        self.model_spec = None
        self.model_handle = None

    async def ListModels(
        self, request: empty_pb2.Empty, context: grpc.aio.ServicerContext
    ) -> nos_service_pb2.ModelListResponse:
        """List all models."""
        return nos_service_pb2.ModelListResponse(models=self.models)

    async def InitModel(
        self, request: nos_service_pb2.InitModelRequest, context: grpc.aio.ServicerContext
    ) -> nos_service_pb2.InitModelResponse:
        """Initialize the model."""
        if self.model_spec and self.model_spec.name == request.model_name:
            return nos_service_pb2.InitModelResponse(result="ok")

        if self.model_spec and self.model_spec.name != request.model_name:
            self.delete_model(self.model_spec.model_name)
            logger.info(f"Resetting model: {request.model_name}")
        logger.info(f"Initializing model: {request.model_name}")

        # Load the model spec
        try:
            self.init_model(request.model_name)
        except Exception as e:
            context.abort(context, grpc.StatusCode.NOT_FOUND, str(e))
        return nos_service_pb2.InitModelResponse(result="ok")

    async def DeleteModel(
        self, request: nos_service_pb2.DeleteModelRequest, context: grpc.aio.ServicerContext
    ) -> nos_service_pb2.DeleteModelResponse:
        """Delete the model."""
        self.delete_model(self.model_spec.name)
        return nos_service_pb2.DeleteModelResponse(result="ok")

    async def Predict(
        self, request: nos_service_pb2.InferenceRequest, context: grpc.aio.ServicerContext
    ) -> nos_service_pb2.InferenceResponse:
        """Main model prediction interface."""
        logger.debug(f"Received request: {request.method}, {request.model_name}")
        if not self.model_spec or not self.model_handle:
            context.abort(context, grpc.StatusCode.FAILED_PRECONDITION, "Model not initialized")

        if self.model_spec.name != request.model_name:
            context.abort(context, grpc.StatusCode.FAILED_PRECONDITION, "Model multiplexing not supported yet")

        # TODO (spillai): This is inconsistent for CLIP which supports both (txt2vec, img2vec)
        # assert self.model_spec.method.value == request.method

        if request.method == MethodType.TXT2IMG.value:
            prompt = request.text_request.text
            logger.debug(f"Generating image with prompt: {prompt}")
            response_ref = self.model_handle.__call__.remote(prompt, height=512, width=512)
            (img,) = await response_ref
            ref_bytes = ray.cloudpickle.dumps({"image": img})
            return nos_service_pb2.InferenceResponse(result=ref_bytes)

        elif request.method == MethodType.TXT2VEC.value:
            prompt = request.text_request.text
            logger.debug(f"Encoding text: {prompt}")
            response_ref = self.model_handle.encode_text.remote(prompt)
            embedding = await response_ref
            ref_bytes = ray.cloudpickle.dumps({"embedding": embedding})
            return nos_service_pb2.InferenceResponse(result=ref_bytes)

        elif request.method == MethodType.IMG2VEC.value:
            img: np.ndarray = ray.cloudpickle.loads(request.image_request.image_bytes)
            logger.debug(f"Encoding img: {img.shape}")

            response_ref = self.model_handle.encode_image.remote(img)
            embedding = await response_ref
            ref_bytes = ray.cloudpickle.dumps({"embedding": embedding})
            return nos_service_pb2.InferenceResponse(result=ref_bytes)
        else:
            context.abort(context, grpc.StatusCode.INVALID_ARGUMENT, f"Invalid method {request.method}")


async def grpc_server(address: str = "[::]:50051") -> None:
    server = grpc.aio.server()
    nos_service_pb2_grpc.add_InferenceServiceServicer_to_server(InferenceService(), server)
    listen_addr = address
    server.add_insecure_port(listen_addr)

    console = rich.console.Console()
    with console.status(f"[bold green] Starting server on {listen_addr}[/bold green]") as status:
        await server.start()
        console.print(
            f"[bold green] ✓ Deployment complete. [/bold green]",  # noqa
        )
        status.stop()
        await server.wait_for_termination()
        console.print("Server stopped")


def main():
    import asyncio

    asyncio.run(grpc_server())


if __name__ == "__main__":
    main()
