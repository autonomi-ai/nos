import time
import traceback
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, Iterator, Union

import grpc
import rich.console
import rich.status
from google.protobuf import empty_pb2, wrappers_pb2

from nos import hub
from nos.common import FunctionSignature, ModelSpec, dumps, loads
from nos.common.shm import NOS_SHM_ENABLED, SharedMemoryDataDict, SharedMemoryTransportManager
from nos.constants import (  # noqa F401
    DEFAULT_GRPC_PORT,  # noqa F401
    GRPC_MAX_MESSAGE_LENGTH,
    GRPC_MAX_WORKER_THREADS,
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
def load_spec(model_id: str) -> ModelSpec:
    """Get the model spec cache."""
    model_spec: ModelSpec = hub.load_spec(model_id)
    logger.info(f"Loaded model spec [name={model_spec.name}]")
    return model_spec


@dataclass
class _StreamingInferenceServiceResponse:
    response: Any

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.response)


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
        self.executor = RayExecutor.get()
        if not self.executor.is_initialized():
            raise RuntimeError("Ray executor is not initialized")
        self.model_manager = ModelManager()
        if NOS_SHM_ENABLED:
            self.shm_manager = SharedMemoryTransportManager()
        else:
            self.shm_manager = None

    def execute(self, model_name: str, method: str = None, inputs: Dict[str, Any] = None, stream: bool = False) -> Any:
        """Execute the model.

        Args:
            model_name (str): Model identifier (e.g. `openai/clip-vit-base-patch32`).
            method (str): Model method to execute (e.g. `__call__`).
            inputs (Dict[str, Any]): Model inputs.
            stream (bool): Whether to stream the response.
        Returns:
            Dict[str, Any]: Model outputs.
        """
        # Load the model spec (with caching to avoid repeated loading)
        try:
            model_spec: ModelSpec = load_spec(model_name)
        except Exception as e:
            raise ModelNotFoundError(f"Failed to load model spec [model_name={model_name}, e={e}]")

        # Validate the model method
        st = time.perf_counter()
        if method is None:
            method = model_spec.default_method
        if method not in model_spec.signature:
            raise ValueError(f"Invalid method [method={method}, model_spec={model_spec}]")

        # Validate inputs based on the model signature
        sig: FunctionSignature = model_spec.signature[method]
        model_inputs: Dict[str, Any] = FunctionSignature.validate(inputs, sig.parameters)
        model_inputs: Dict[str, Any] = SharedMemoryDataDict.decode(model_inputs)

        # Profile the inputs if enabled
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
        model_handle: ModelHandle = self.model_manager.load(model_spec)

        st = time.perf_counter()
        if not stream:
            # Get the model handle and call it remotely (with model spec, actor handle)
            response: Union[Any, Dict[str, Any]] = model_handle(**model_inputs, _method=method)
            if NOS_PROFILING_ENABLED:
                logger.debug(
                    f"Executed model [name={model_spec.name}, elapsed={(time.perf_counter() - st) * 1e3:.1f}ms]"
                )

            # If the response is a single value, wrap it in a dict with the appropriate key
            if isinstance(sig.output_annotations, dict) and len(sig.output_annotations) == 1:
                response = {k: response for k in sig.output_annotations}
            return response
        else:
            response: Iterator[Any] = model_handle(**model_inputs, _method=method, _stream=True)
            return _StreamingInferenceServiceResponse(response)


class InferenceServiceImpl(nos_service_pb2_grpc.InferenceServiceServicer, InferenceService):
    """
    Experimental gRPC-based inference service.

    This service is used to serve models over gRPC.

    Refer to the bring-your-own-schema section:
    https://docs.ray.io/en/master/serve/direct-ingress.html?highlight=grpc#bring-your-own-schema
    """

    def __init__(self, *args, **kwargs):
        self.executor = RayExecutor.get()
        try:
            self.executor.init()
        except Exception as e:
            err_msg = f"Failed to initialize executor [e={e}]"
            logger.info(err_msg)
            raise RuntimeError(err_msg)
        self._tmp_files = {}
        super().__init__(*args, **kwargs)

    def Ping(self, _: empty_pb2.Empty, context: grpc.ServicerContext) -> nos_service_pb2.PingResponse:
        """Health check."""
        return nos_service_pb2.PingResponse(status="ok")

    def GetServiceInfo(self, _: empty_pb2.Empty, context: grpc.ServicerContext) -> nos_service_pb2.ServiceInfoResponse:
        """Get information on the service."""
        from nos.common.system import has_gpu, is_inside_docker

        if is_inside_docker():
            runtime = "gpu" if has_gpu() else "cpu"
        else:
            runtime = "local"
        return nos_service_pb2.ServiceInfoResponse(version=__version__, runtime=runtime)

    def ListModels(self, _: empty_pb2.Empty, context: grpc.ServicerContext) -> nos_service_pb2.GenericResponse:
        """List all models."""
        models = list(hub.list())
        logger.debug(f"ListModels() [models={len(models)}]")
        return nos_service_pb2.GenericResponse(response_bytes=dumps(models))

    def GetModelInfo(
        self, request: wrappers_pb2.StringValue, context: grpc.ServicerContext
    ) -> nos_service_pb2.GenericResponse:
        """Get model information."""
        try:
            model_id = request.value
            spec: ModelSpec = hub.load_spec(model_id)
            logger.debug(f"GetModelInfo() [spec={spec}]")
        except KeyError as e:
            logger.error(f"Failed to load spec [request={request}, e={e}]")
            context.abort(grpc.StatusCode.NOT_FOUND, str(e))
        return spec._to_proto()

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

    def UploadFile(self, request_iterator: Any, context: grpc.ServicerContext) -> nos_service_pb2.GenericResponse:
        """Upload a file."""
        for _chunk_idx, chunk_request in enumerate(request_iterator):
            chunk = loads(chunk_request.request_bytes)
            chunk_bytes = chunk["chunk_bytes"]
            path = Path(chunk["filename"]).absolute()
            if str(path) not in self._tmp_files:
                tmp_file = NamedTemporaryFile(delete=False, dir="/tmp", suffix=path.suffix)
                self._tmp_files[str(path)] = tmp_file
                logger.debug(
                    f"Streaming upload [path={tmp_file.name}, size={Path(tmp_file.name).stat().st_size / (1024 * 1024):.2f} MB]"
                )
            else:
                tmp_file = self._tmp_files[str(path)]
            with open(tmp_file.name, "ab") as f:
                f.write(chunk_bytes)
        return nos_service_pb2.GenericResponse(response_bytes=dumps({"filename": tmp_file.name}))

    def DeleteFile(self, request: nos_service_pb2.GenericRequest, context: grpc.ServicerContext) -> empty_pb2.Empty:
        """Delete a file by its file-identifier."""
        request = loads(request.request_bytes)

        filename = str(request["filename"])
        try:
            tmp_file = self._tmp_files[str(filename)]
            path = Path(tmp_file.name)
            assert path.exists(), f"File handle {filename} not found"
        except Exception as e:
            err_msg = f"Failed to delete file [filename={filename}, e={e}]"
            logger.error(err_msg)
            context.abort(grpc.StatusCode.NOT_FOUND, err_msg)

        logger.debug(f"Deleting file [path={path}]")
        path.unlink()
        del self._tmp_files[str(filename)]
        return empty_pb2.Empty()

    def Run(
        self, request: nos_service_pb2.GenericRequest, context: grpc.ServicerContext
    ) -> nos_service_pb2.GenericResponse:
        """Main model prediction interface."""
        request: Dict[str, Any] = loads(request.request_bytes)
        try:
            st = time.perf_counter()
            logger.info(f"Executing request [model={request['id']}, method={request['method']}]")
            response = self.execute(request["id"], method=request["method"], inputs=request["inputs"])
            logger.info(
                f"Executed request [model={request['id']}, method={request['method']}, elapsed={(time.perf_counter() - st) * 1e3:.1f}ms]"
            )
            return nos_service_pb2.GenericResponse(response_bytes=dumps(response))
        except (grpc.RpcError, Exception) as e:
            msg = f"Failed to execute request [model={request['id']}, method={request['method']}]"
            msg += f"{traceback.format_exc()}"
            logger.error(f"{msg}, e={e}")
            context.abort(grpc.StatusCode.INTERNAL, "Internal Server Error")

    def Stream(
        self, request: nos_service_pb2.GenericRequest, context: grpc.ServicerContext
    ) -> Iterator[nos_service_pb2.GenericResponse]:
        """Main streaming model prediction interface."""
        request: Dict[str, Any] = loads(request.request_bytes)
        try:
            logger.info(f"Executing request [model={request['id']}, method={request['method']}]")
            for response in self.execute(
                request["id"], method=request["method"], inputs=request["inputs"], stream=True
            ):
                yield nos_service_pb2.GenericResponse(response_bytes=dumps(response))
            logger.info(f"Executed request [model={request['id']}, method={request['method']}]")
        except (grpc.RpcError, Exception) as e:
            msg = f"Failed to execute request [model={request['id']}, method={request['method']}]"
            msg += f"{traceback.format_exc()}"
            logger.error(f"{msg}, e={e}")
            context.abort(grpc.StatusCode.INTERNAL, "Internal Server Error")


def serve(address: str = f"[::]:{DEFAULT_GRPC_PORT}", max_workers: int = GRPC_MAX_WORKER_THREADS) -> None:
    """Start the gRPC server."""
    from concurrent import futures

    options = [
        ("grpc.max_message_length", GRPC_MAX_MESSAGE_LENGTH),
        ("grpc.max_send_message_length", GRPC_MAX_MESSAGE_LENGTH),
        ("grpc.max_receive_message_length", GRPC_MAX_MESSAGE_LENGTH),
    ]
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers), options=options)
    nos_service_pb2_grpc.add_InferenceServiceServicer_to_server(InferenceServiceImpl(), server)
    server.add_insecure_port(address)

    console = rich.console.Console()
    console.print(f"[bold green] ✓ Starting gRPC server on {address}[/bold green]")
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
