import ray
import rich.console
import rich.status
from google.protobuf import empty_pb2
from ray import serve

import grpc
from nos import hub
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

    async def ListModels(
        self, request: empty_pb2.Empty, context: grpc.aio.ServicerContext
    ) -> nos_service_pb2.ModelListResponse:
        return nos_service_pb2.ModelListResponse(models=self.models)

    async def InitModel(
        self, request: nos_service_pb2.InitModelRequest, context: grpc.aio.ServicerContext
    ) -> nos_service_pb2.InitModelResponse:
        if self.model_spec and self.model_spec.name == request.model_name:
            return nos_service_pb2.InitModelResponse(result="ok")

        if self.model_spec and self.model_spec.name != request.model_name:
            raise grpc.aio.ServicerContext.abort(
                context, grpc.StatusCode.FAILED_PRECONDITION, "Model multiplexing not supported yet"
            )
        logger.info(f"Initializing model: {request.model_name}")

        # Load the model spec
        try:
            self.model_spec = hub.load_spec(request.model_name)
            logger.info(f"Loaded model spec: {self.model_spec}")
        except Exception as e:
            raise grpc.aio.ServicerContext.abort(context, grpc.StatusCode.NOT_FOUND, str(e))

        # Create the serve deployment from the model handle
        model_cls = self.model_spec.cls
        deployment_config = {
            "ray_actor_options": {"num_gpus": 1},  # TODO (spillai): this should be available from the model spec
            "autoscaling_config": {"min_replicas": request.min_replicas, "max_replicas": request.max_replicas},
        }

        # Build the ray deployment handle
        logger.info(f"Creating deployment with config: {deployment_config}")
        deployment = serve.deployment(**deployment_config)(model_cls)
        logger.info(f"Created deployment: {deployment}")
        self.handle = deployment.bind(*self.model_spec.args, **self.model_spec.kwargs).execute()
        logger.info(f"Bound deployment: {self.handle}, type={type(self.handle)}")
        return nos_service_pb2.InitModelResponse(result="ok")

    async def Predict(
        self, request: nos_service_pb2.InferenceRequest, context: grpc.aio.ServicerContext
    ) -> nos_service_pb2.InferenceResponse:
        logger.debug(f"Received request: {request.method}, {request.model_name}")
        if not self.model_spec or not self.handle:
            raise grpc.aio.ServicerContext.abort(context, grpc.StatusCode.FAILED_PRECONDITION, "Model not initialized")

        if self.model_spec.name != request.model_name:
            raise grpc.aio.ServicerContext.abort(
                context, grpc.StatusCode.FAILED_PRECONDITION, "Model multiplexing not supported yet"
            )
        assert self.model_spec.method.value == request.method

        if request.method == MethodType.TXT2IMG.value:
            logger.debug(f"Generating image with prompt: {request.text}")
            response_ref = self.handle.__call__.remote(request.text, height=512, width=512)
            (img,) = await response_ref
            img_ref = ray.put(img)
            ref_bytes = ray.cloudpickle.dumps(img_ref)
            # file_stream = BytesIO()
            # img.save(file_stream, "PNG")
            return nos_service_pb2.InferenceResponse(result=ref_bytes)

        elif request.method == MethodType.TXT2VEC.value:
            logger.debug(f"Encoding text: {request.text}")
            response_ref = self.handle.encode_text.remote(request.text)
            embedding = await response_ref
            embedding_ref = ray.put({"embedding": embedding})
            ref_bytes = ray.cloudpickle.dumps(embedding_ref)
            return nos_service_pb2.InferenceResponse(result=ref_bytes)

        else:
            raise grpc.aio.ServicerContext.abort(
                context, grpc.StatusCode.INVALID_ARGUMENT, f"Invalid method {request.method}"
            )


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


if __name__ == "__main__":
    import asyncio

    asyncio.run(grpc_server())
