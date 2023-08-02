import time
import traceback
from functools import lru_cache
from typing import Any, Dict

import grpc
import rich.console
import rich.status
from google.protobuf import empty_pb2

from nos import hub
from nos.common import FunctionSignature, ModelSpec, TaskType, dumps, loads
from nos.common.shm import NOS_SHM_ENABLED, SharedMemoryDataDict, SharedMemoryTransportManager
from nos.constants import (  # noqa F401
    DEFAULT_GRPC_PORT,  # noqa F401
    NOS_PROFILING_ENABLED,
)
from nos.exceptions import ModelNotFoundError
from nos.executors.ray import RayExecutor
from nos.logging import logger
from nos.managers import ModelHandle, ModelManager
from nos.protoc import import_module
from nos.version import __version__


nos_service_pb2 = import_module("nos_service_pb2")
nos_service_pb2_grpc = import_module("nos_service_pb2_grpc")


@lru_cache(maxsize=32)
def load_spec(model_name: str, task: TaskType) -> ModelSpec:
    """Get the model spec cache."""
    model_spec: ModelSpec = hub.load_spec(model_name, task=task)
    logger.info(f"Loaded model spec [task={model_spec.task.value}, name={model_spec.name}]")
    return model_spec


class InferenceService:
    """Ray-executor based inference service.

    Parameters:
        model_manager (ModelManager): Model manager.
        executor (RayExecutor): Ray executor.
        shm_manager (SharedMemoryTransportManager): Shared memory transport manager.
            Used to create shared memory buffers for inputs/outputs,
            and to copy data to/from shared memory.

    Note: To be used with the `InferenceServiceImpl` gRPC service.
    """

    def __init__(self):
        self.model_manager = ModelManager()
        self.executor = RayExecutor.get()
        try:
            self.executor.init()
        except Exception as e:
            err_msg = f"Failed to initialize executor [e={e}]"
            logger.info(err_msg)
            raise RuntimeError(err_msg)
        if NOS_SHM_ENABLED:
            self.shm_manager = SharedMemoryTransportManager()
        else:
            self.shm_manager = None

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
            model_spec: ModelSpec = load_spec(model_name, task=task)
        except Exception as e:
            raise ModelNotFoundError(f"Failed to load model spec [model_name={model_name}, e={e}]")

        # TODO (spillai): Validate/Decode the inputs
        st = time.perf_counter()
        model_inputs = FunctionSignature.validate(inputs, model_spec.signature.inputs)
        model_inputs = SharedMemoryDataDict.decode(model_inputs)
        if NOS_PROFILING_ENABLED:
            model_inputs_types = [
                f"{k}: List[type={type(v[0])}, len={len(v)}]" if isinstance(v, list) else str(type(v))
                for k, v in model_inputs.items()
            ]
            logger.debug(
                f"Decoded inputs [inputs=({', '.join(model_inputs_types)}), elapsed={(time.perf_counter() - st) * 1e3:.1f}ms]"
            )

        # Initialize the model (if not already initialized)
        # This call should also evict models and garbage collect if
        # too many models are loaded are loaded simultaneously.
        model_handle: ModelHandle = self.model_manager.get(model_spec)

        # Get the model handle and call it remotely (with model spec, actor handle)
        st = time.perf_counter()
        response: Dict[str, Any] = model_handle(**model_inputs)
        if NOS_PROFILING_ENABLED:
            logger.debug(f"Executed model [name={model_spec.name}, elapsed={(time.perf_counter() - st) * 1e3:.1f}ms]")

        # If the response is a single value, wrap it in a dict with the appropriate key
        if len(model_spec.signature.outputs) == 1:
            response = {k: response for k in model_spec.signature.outputs}

        # Encode the response
        st = time.perf_counter()
        response = SharedMemoryDataDict.encode(response)
        if NOS_PROFILING_ENABLED:
            logger.debug(f"Encoded response [elapsed={(time.perf_counter() - st) * 1e3:.1f}ms]")

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
            logger.debug(f"GetModelInfo() [spec={spec}]")
        except KeyError as e:
            logger.error(f"Failed to load spec [request={request.request}, e={e}]")
            context.abort(grpc.StatusCode.NOT_FOUND, str(e))
        return spec._to_proto(public=True)

    def RegisterSystemSharedMemory(
        self, request: nos_service_pb2.GenericRequest, context: grpc.ServicerContext
    ) -> nos_service_pb2.GenericResponse:
        """Register system shared memory under a specific namespace `<client_id>/<object_id>`."""
        if not NOS_SHM_ENABLED:
            context.abort(grpc.StatusCode.UNIMPLEMENTED, "Shared memory not enabled.")

        metadata = dict(context.invocation_metadata())
        client_id = metadata.get("client_id", None)
        object_id = metadata.get("object_id", None)
        namespace = f"{client_id}/{object_id}"
        logger.debug(f"Registering shm [client_id={client_id}, object_id={object_id}]")
        try:
            # Create a shared memory segment for the inputs
            # Note: The returned keys for shared memory segments are identical to the
            # keys in the input dictionary (i.e. <key>), and are not prefixed with the
            # namespace `<client_id>/<object_id>`.
            shm_map = self.shm_manager.create(loads(request.request_bytes), namespace=namespace)
            # Here, dumps() is used to serialize the shared memory numy objects via __getstate__().
            # The serialized data is then sent back to the client, which can then deserialized
            # and set via __setstate__() on the client-side, so that the client can access the shared
            # memory segments.
            logger.debug(f"Registered shm [client_id={client_id}, object_id={object_id}, shm_map={shm_map}]")
            return nos_service_pb2.GenericResponse(response_bytes=dumps(shm_map))
        except Exception as e:
            logger.error(f"Failed to register system shared memory: {e}")
            context.abort(grpc.StatusCode.INTERNAL, str(e))

    def UnregisterSystemSharedMemory(
        self, request: nos_service_pb2.GenericRequest, context: grpc.ServicerContext
    ) -> nos_service_pb2.GenericResponse:
        """Unregister system shared memory for specific namespace `<client_id>/<object_id>`."""
        if not NOS_SHM_ENABLED:
            context.abort(context, grpc.StatusCode.UNIMPLEMENTED, "Shared memory not enabled.")

        metadata = dict(context.invocation_metadata())
        client_id = metadata.get("client_id", None)
        object_id = metadata.get("object_id", None)
        namespace = f"{client_id}/{object_id}"
        # TODO (spillai): Currently, we can ignore the `request` provided
        # by the client, since all the shared memory segments under the namespace are deleted.
        logger.debug(f"Unregistering shm [client_id={client_id}, object_id={object_id}]")
        try:
            self.shm_manager.cleanup(namespace=namespace)
        except Exception as e:
            logger.error(f"Failed to unregister shm [e{e}]")
            context.abort(grpc.StatusCode.INTERNAL, str(e))
        return nos_service_pb2.GenericResponse()

    def Run(
        self, request: nos_service_pb2.InferenceRequest, context: grpc.ServicerContext
    ) -> nos_service_pb2.InferenceResponse:
        """Main model prediction interface."""
        model_request = request.model
        logger.debug(f"=> Received request [task={model_request.task}, model={model_request.name}]")
        if model_request.task not in (
            TaskType.IMAGE_GENERATION.value,
            TaskType.IMAGE_EMBEDDING.value,
            TaskType.TEXT_EMBEDDING.value,
            TaskType.OBJECT_DETECTION_2D.value,
            TaskType.IMAGE_SEGMENTATION_2D.value,
            TaskType.CUSTOM.value,
        ):
            context.abort(grpc.StatusCode.NOT_FOUND, f"Invalid task [task={model_request.task}]")

        try:
            st = time.perf_counter()
            logger.info(f"Executing request [task={model_request.task}, model={model_request.name}]")
            response = self.execute(model_request.name, task=TaskType(model_request.task), inputs=request.inputs)
            logger.info(
                f"Executed request [task={model_request.task}, model={model_request.name}, elapsed={(time.perf_counter() - st) * 1e3:.1f}ms]"
            )
            return nos_service_pb2.InferenceResponse(response_bytes=dumps(response))
        except (grpc.RpcError, Exception) as e:
            msg = f"Failed to execute request [task={model_request.task}, model={model_request.name}]"
            msg += f"{traceback.format_exc()}"
            logger.error(f"{msg}, e={e}")
            context.abort(grpc.StatusCode.INTERNAL, "Internal Server Error")


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
    console.print(f"[bold green] Starting server on {address}[/bold green]")
    start_t = time.time()
    server.start()
    console.print(
        f"[bold green] ✓ InferenceService :: Deployment complete (elapsed={time.time() - start_t:.1f}s) [/bold green]",  # noqa
    )
    server.wait_for_termination()
    console.print("Server stopped")


def main():
    serve()


if __name__ == "__main__":
    main()
