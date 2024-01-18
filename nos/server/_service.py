import time
import traceback
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, Iterator, List, Union

import grpc
import rich.console
import rich.status
from google.protobuf import empty_pb2, wrappers_pb2

from nos import hub
from nos.common import (
    FunctionSignature,
    ModelDeploymentSpec,
    ModelServiceSpec,
    ModelSpec,
    ModelSpecMetadataCatalog,
    dumps,
    loads,
)
from nos.common.shm import NOS_SHM_ENABLED, SharedMemoryDataDict, SharedMemoryTransportManager
from nos.constants import (  # noqa F401
    DEFAULT_GRPC_ADDRESS,
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
    logger.debug(f"Loaded model spec [name={model_spec.name}]")
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
        # Ray executor to execute models
        self.executor = RayExecutor.get()
        try:
            self.executor.init()
        except Exception as e:
            err_msg = f"Failed to initialize executor [e={e}]"
            logger.error(err_msg)
            raise RuntimeError(err_msg)
        # Model manager to manage model loading / unloading
        self.model_manager = ModelManager()
        # Shared memory transport manager for faster IPC
        if NOS_SHM_ENABLED:
            self.shm_manager = SharedMemoryTransportManager()
        else:
            self.shm_manager = None

    def load_model_spec(self, spec: ModelSpec, deployment: ModelDeploymentSpec) -> ModelHandle:
        """Load the model by spec."""
        return self.model_manager.load(spec, deployment)

    def load_model(self, model_name: str, num_replicas: int = 1) -> ModelHandle:
        """Load the model by model name."""
        # Load the model spec (with caching to avoid repeated loading)
        try:
            spec: ModelSpec = load_spec(model_name)
        except Exception as e:
            raise ModelNotFoundError(f"Failed to load model spec [model_name={model_name}, e={e}]")
        return self.load_model_spec(spec, ModelDeploymentSpec(num_replicas=num_replicas))

    async def execute_model(
        self, model_name: str, method: str = None, inputs: Dict[str, Any] = None, stream: bool = False
    ) -> Any:
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
        # Note: if the model hasn't been loaded yet, then this call will
        # block until the model is loaded (num_replicas=1).
        model_handle: ModelHandle = self.model_manager.get(model_spec)

        st = time.perf_counter()
        if not stream:
            # Get the model handle and call it remotely (with model spec, actor handle)
            if model_handle.num_replicas > 1:
                # If the model has multiple replicas, then call the submit method
                response_ref = model_handle.submit(**model_inputs, _method=method)
                logger.debug(
                    f"Submitted model request, awaiting response [handle={model_handle}, response_ref={response_ref}]"
                )
                st = time.time()
                response: Union[Any, Dict[str, Any]] = await model_handle.async_get(response_ref)
                logger.debug(
                    f"Response awaited [handle={model_handle}, response={response}, elapsed={(time.time() - st) * 1e3:.1f}ms]"
                )
            else:
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
            if model_handle.num_replicas > 1:
                # If the model has multiple replicas, then call the submit method
                response_ref: Iterator[Any] = model_handle.submit(**model_inputs, _method=method, _stream=True)
                response: Iterator[Any] = await model_handle.async_get(response_ref)
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

    def __init__(self, catalog_filename: str = None):
        super().__init__()
        self._tmp_files = {}

        if catalog_filename is None:
            return

        if not Path(catalog_filename).exists():
            raise ValueError(f"Model catalog not found [catalog={catalog_filename}]")

        # Register models from the catalog
        services: List[ModelServiceSpec] = hub.register_from_yaml(catalog_filename)
        for svc in services:
            logger.debug(f"Servicing model [svc={svc}, replicas={svc.deployment.num_replicas}]")
            self.load_model_spec(svc.model, svc.deployment)
            logger.debug(f"Deployed model [svc={svc}]. \n{self.model_manager}")

    async def Ping(self, _: empty_pb2.Empty, context: grpc.ServicerContext) -> nos_service_pb2.PingResponse:
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

    def GetModelCatalog(self, _: empty_pb2.Empty, context: grpc.ServicerContext) -> nos_service_pb2.GenericResponse:
        """Get the model catalog."""
        catalog = ModelSpecMetadataCatalog.get()
        logger.debug(f"GetModelCatalog() [catalog={catalog._metadata_catalog}]")
        return nos_service_pb2.GenericResponse(response_bytes=dumps(catalog))

    def GetModelInfo(
        self, request: wrappers_pb2.StringValue, context: grpc.ServicerContext
    ) -> nos_service_pb2.GenericResponse:
        """Get model information."""
        try:
            model_id = request.value
            spec: ModelSpec = hub.load_spec(model_id)
        except KeyError as e:
            logger.error(f"Failed to load spec [request={request}, e={e}]")
            context.abort(grpc.StatusCode.NOT_FOUND, str(e))
        return spec._to_proto()

    def LoadModel(self, request: nos_service_pb2.GenericRequest, context: grpc.ServicerContext) -> empty_pb2.Empty:
        """Load / scale the model to the specified number of replicas."""
        request: Dict[str, Any] = loads(request.request_bytes)
        logger.debug(f"ScaleModel() [request={request}]")
        try:
            model_id = request["id"]
            num_replicas = request.get("num_replicas", 1)
            self.load_model(model_id, num_replicas=num_replicas)
            return empty_pb2.Empty()
        except Exception as e:
            err_msg = f"Failed to scale model [model_id={model_id}, num_replicas={num_replicas}, e={e}]"
            logger.error(err_msg)
            context.abort(grpc.StatusCode.INTERNAL, err_msg)

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

    async def Run(
        self, request: nos_service_pb2.GenericRequest, context: grpc.ServicerContext
    ) -> nos_service_pb2.GenericResponse:
        """Main model prediction interface."""
        request: Dict[str, Any] = loads(request.request_bytes)
        try:
            st = time.perf_counter()
            logger.info(f"Executing request [model={request['id']}, method={request['method']}]")
            response = await self.execute_model(request["id"], method=request["method"], inputs=request["inputs"])
            logger.info(
                f"Executed request [model={request['id']}, method={request['method']}, elapsed={(time.perf_counter() - st) * 1e3:.1f}ms]"
            )
            return nos_service_pb2.GenericResponse(response_bytes=dumps(response))
        except (grpc.RpcError, Exception) as e:
            msg = f"Failed to execute request [model={request['id']}, method={request['method']}]"
            msg += f"{traceback.format_exc()}"
            logger.error(f"{msg}, e={e}")
            context.abort(grpc.StatusCode.INTERNAL, "Internal Server Error")

    async def Stream(
        self, request: nos_service_pb2.GenericRequest, context: grpc.ServicerContext
    ) -> Iterator[nos_service_pb2.GenericResponse]:
        """Main streaming model prediction interface."""
        request: Dict[str, Any] = loads(request.request_bytes)
        try:
            logger.info(f"Executing request [model={request['id']}, method={request['method']}]")
            response_stream = await self.execute_model(
                request["id"], method=request["method"], inputs=request["inputs"], stream=True
            )
            for response in response_stream:
                yield nos_service_pb2.GenericResponse(response_bytes=dumps(response))
            logger.info(f"Executed request [model={request['id']}, method={request['method']}]")
        except (grpc.RpcError, Exception) as e:
            msg = f"Failed to execute request [model={request['id']}, method={request['method']}]"
            msg += f"{traceback.format_exc()}"
            logger.error(f"{msg}, e={e}")
            context.abort(grpc.StatusCode.INTERNAL, "Internal Server Error")


async def async_serve_impl(
    address: str = DEFAULT_GRPC_ADDRESS,
    wait_for_termination: bool = True,
    catalog: str = None,
):
    from grpc import aio

    options = [
        ("grpc.max_message_length", GRPC_MAX_MESSAGE_LENGTH),
        ("grpc.max_send_message_length", GRPC_MAX_MESSAGE_LENGTH),
        ("grpc.max_receive_message_length", GRPC_MAX_MESSAGE_LENGTH),
    ]
    server = aio.server(options=options)
    nos_service_pb2_grpc.add_InferenceServiceServicer_to_server(InferenceServiceImpl(catalog), server)
    server.add_insecure_port(address)

    console = rich.console.Console()
    console.print(f"[bold green] ✓ Starting gRPC server on {address}[/bold green]")

    start_t = time.time()
    logger.debug(f"Starting gRPC server on {address}")
    await server.start()
    console.print(
        f"[bold green] ✓ InferenceService :: Deployment complete (elapsed={time.time() - start_t:.1f}s) [/bold green]",  # noqa
    )
    if not wait_for_termination:
        return server
    try:
        logger.debug("Waiting for server termination")
        await server.wait_for_termination()
    except KeyboardInterrupt:
        logger.debug("Received KeyboardInterrupt, stopping server")
        await server.stop(0)
        logger.debug("Server stopped")
        console.print("[bold green] ✓ InferenceService :: Server stopped. [/bold green]")
    return server


def async_serve(
    address: str = DEFAULT_GRPC_ADDRESS,
    max_workers: int = GRPC_MAX_WORKER_THREADS,
    wait_for_termination: bool = True,
    catalog: str = None,
):
    """Start the gRPC server."""
    import asyncio

    loop = asyncio.new_event_loop()
    task = loop.create_task(async_serve_impl(address, wait_for_termination, catalog))
    loop.run_until_complete(task)
    return task.result()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Inference service")
    parser.add_argument("-a", "--address", type=str, default=DEFAULT_GRPC_ADDRESS, help="gRPC server address")
    parser.add_argument("-c", "--catalog", type=str, default=None, help="Model catalog")
    args = parser.parse_args()
    logger.debug(f"args={args}")

    async_serve(address=args.address, catalog=args.catalog)


if __name__ == "__main__":
    main()
