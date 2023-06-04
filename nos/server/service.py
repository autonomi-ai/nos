from typing import Any, Dict

import grpc
import rich.console
import rich.status
from google.protobuf import empty_pb2

from nos import hub
from nos.common import FunctionSignature, ModelSpec, TaskType, dumps, loads
from nos.common.shm import NOS_SHM_ENABLED, SharedMemoryDataDict, SharedMemoryTransportManager
from nos.constants import DEFAULT_GRPC_PORT  # noqa F401
from nos.exceptions import ModelNotFoundError
from nos.executors.ray import RayExecutor
from nos.logging import logger
from nos.managers import ModelHandle, ModelManager
from nos.protoc import import_module
from nos.version import __version__


nos_service_pb2 = import_module("nos_service_pb2")
nos_service_pb2_grpc = import_module("nos_service_pb2_grpc")


class InferenceService:
    """Ray-executor based inference service.

    Parameters:
        model_manager (ModelManager): Model manager.
        executor (RayExecutor): Ray executor.
        shm_manager (SharedMemoryTransportManager): Shared memory transport manager.
            Used to create shared memory buffers for inputs/outputs,
            and to copy data to/from shared memory.
    """

    def __init__(self):
        self.model_manager = ModelManager()
        self.executor = RayExecutor.get()
        try:
            self.executor.init()
        except Exception as e:
            logger.info(f"Failed to initialize executor: {e}")
            raise RuntimeError(f"Failed to initialize executor: {e}")
        if NOS_SHM_ENABLED:
            self.shm_manager = SharedMemoryTransportManager()
        else:
            self.shm_manager = None

    @logger.catch
    def execute(self, model_name: str, task: TaskType = None, inputs: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute the model.

        Args:
            model_name (str): Model identifier (e.g. `openai/clip-vit-base-patch32`).
            task (TaskType): Task type (e.g. `TaskType.OBJECT_DETECTION_2D`).
            inputs (Dict[str, Any]): Model inputs.
        Returns:
            Dict[str, Any]: Model outputs.
        """
        # Load the model spec
        try:
            model_spec: ModelSpec = hub.load_spec(model_name, task=task)
            logger.debug(f"Loaded model spec: {model_spec}")
        except Exception as e:
            raise ModelNotFoundError(f"Failed to load model spec: {model_name}, {e}")

        # TODO (spillai): Validate/Decode the inputs
        model_inputs = FunctionSignature.validate(inputs, model_spec.signature.inputs)
        model_inputs = SharedMemoryDataDict.decode(model_inputs)

        # Initialize the model (if not already initialized)
        # This call should also evict models and garbage collect if
        # too many models are loaded are loaded simultaneously.
        model_handle: ModelHandle = self.model_manager.get(model_spec)

        # Get the model handle and call it remotely (with model spec, actor handle)
        response: Dict[str, Any] = model_handle.remote(**model_inputs)

        # If the response is a single value, wrap it in a dict with the appropriate key
        if len(model_spec.signature.outputs) == 1:
            response: Dict[str, Any] = {k: response for k in model_spec.signature.outputs}

        # Encode the response
        response = SharedMemoryDataDict.encode(response)

        return response


class InferenceServiceImpl(nos_service_pb2_grpc.InferenceServiceServicer, InferenceService):
    """
    Experimental gRPC-based inference service.

    This service is used to serve models over gRPC.

    Refer to the bring-your-own-schema section:
    https://docs.ray.io/en/master/serve/direct-ingress.html?highlight=grpc#bring-your-own-schema
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def Ping(self, request: empty_pb2.Empty, context: grpc.ServicerContext) -> nos_service_pb2.PingResponse:
        """Health check."""
        return nos_service_pb2.PingResponse(status="ok")

    def GetServiceInfo(
        self, request: empty_pb2.Empty, context: grpc.ServicerContext
    ) -> nos_service_pb2.ServiceInfoResponse:
        """Get information on the service."""
        return nos_service_pb2.ServiceInfoResponse(version=__version__)

    def ListModels(self, request: empty_pb2.Empty, context: grpc.ServicerContext) -> nos_service_pb2.ModelListResponse:
        """List all models."""
        response = nos_service_pb2.ModelListResponse()
        for spec in hub.list():
            response.models.append(nos_service_pb2.ModelInfo(name=spec.name, task=spec.task.value))
        return response

    def GetModelInfo(
        self, request: nos_service_pb2.ModelInfoRequest, context: grpc.ServicerContext
    ) -> nos_service_pb2.ModelInfoResponse:
        """Get model information."""
        try:
            model_info = request.request
            spec: ModelSpec = hub.load_spec(model_info.name, task=TaskType(model_info.task))
        except KeyError as e:
            logger.error(f"Model not found: {e}")
            context.abort(context, grpc.StatusCode.NOT_FOUND, str(e))
        return spec._to_proto(public=True)

    def RegisterSystemSharedMemory(
        self, request: nos_service_pb2.GenericRequest, context: grpc.ServicerContext
    ) -> nos_service_pb2.GenericResponse:
        """Register system shared memory with a dry-run inference request."""
        if not NOS_SHM_ENABLED:
            context.abort(context, grpc.StatusCode.UNIMPLEMENTED, "Shared memory not enabled.")

        metadata = dict(context.invocation_metadata())
        client_id = metadata.get("client_id", None)
        spec_id = metadata.get("spec_id", None)
        logger.debug(f"Registering system shared memory for client_id={client_id}, spec_id={spec_id}")
        try:
            shm_map = self.shm_manager.create(loads(request.request_bytes), namespace=f"{client_id}/{spec_id}")
            return nos_service_pb2.GenericResponse(response_bytes=dumps(shm_map))
        except Exception as e:
            logger.error(f"Failed to register system shared memory: {e}")
            context.abort(context, grpc.StatusCode.INTERNAL, str(e))

    def UnregisterSystemSharedMemory(
        self, request: nos_service_pb2.GenericRequest, context: grpc.ServicerContext
    ) -> nos_service_pb2.GenericResponse:
        """Unregister system shared memory."""
        if not NOS_SHM_ENABLED:
            context.abort(context, grpc.StatusCode.UNIMPLEMENTED, "Shared memory not enabled.")

        metadata = dict(context.invocation_metadata())
        client_id = metadata.get("client_id", None)
        spec_id = metadata.get("spec_id", None)
        logger.debug(f"Unregistering system shared memory for client_id={client_id}, spec_id={spec_id}")
        try:
            self.shm_manager.cleanup(namespace=f"{client_id}/{spec_id}")
        except Exception as e:
            logger.error(f"Failed to unregister system shared memory: {e}")
            context.abort(context, grpc.StatusCode.INTERNAL, str(e))
        return nos_service_pb2.GenericResponse()

    def Run(
        self, request: nos_service_pb2.InferenceRequest, context: grpc.ServicerContext
    ) -> nos_service_pb2.InferenceResponse:
        """Main model prediction interface."""
        model_request = request.model
        logger.debug(f"Received request: {model_request.task}, {model_request.name}")
        if model_request.task not in (
            TaskType.IMAGE_GENERATION.value,
            TaskType.IMAGE_EMBEDDING.value,
            TaskType.TEXT_EMBEDDING.value,
            TaskType.OBJECT_DETECTION_2D.value,
            TaskType.IMAGE_SEGMENTATION_2D.value,
        ):
            context.abort(context, grpc.StatusCode.NOT_FOUND, f"Invalid task {model_request.task}")

        try:
            logger.debug(f"Executing request: {model_request.task}, {model_request.name}")
            response = self.execute(model_request.name, task=TaskType(model_request.task), inputs=request.inputs)
            return nos_service_pb2.InferenceResponse(response_bytes=dumps(response))
        except Exception as e:
            logger.error(f"Failed to execute request: {model_request.task}, {model_request.name}, {request.inputs}")
            context.abort(context, grpc.StatusCode.INTERNAL, str(e))


def serve(address: str = f"[::]:{DEFAULT_GRPC_PORT}", max_workers: int = 1) -> None:
    """Start the gRPC server."""
    from concurrent import futures

    options = [
        ("grpc.max_message_length", 512 * 1024 * 1024),
        ("grpc.max_send_message_length", 512 * 1024 * 1024),
        ("grpc.max_receive_message_length", 512 * 1024 * 1024),
    ]
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers), options=options)
    nos_service_pb2_grpc.add_InferenceServiceServicer_to_server(InferenceServiceImpl(), server)
    server.add_insecure_port(address)

    console = rich.console.Console()
    with console.status(f"[bold green] Starting server on {address}[/bold green]") as status:
        server.start()
        console.print(
            f"[bold green] ✓ InferenceService :: Deployment complete [/bold green]",  # noqa
        )
        status.stop()
        server.wait_for_termination()
        console.print("Server stopped")


def main():
    serve()


if __name__ == "__main__":
    main()
