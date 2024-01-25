"""gRPC client for NOS service."""
import contextlib
import secrets
import time
from dataclasses import dataclass, field
from functools import cached_property, lru_cache, partial
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List

import grpc
import numpy as np
from google.protobuf import empty_pb2, wrappers_pb2
from PIL import Image

from nos.common import FunctionSignature, ModelSpec, ModelSpecMetadataCatalog, TensorSpec, dumps, loads
from nos.common.exceptions import (
    ClientException,
    InferenceException,
    InputValidationException,
    ServerReadyException,
)
from nos.common.shm import NOS_SHM_ENABLED, SharedMemoryTransportManager
from nos.constants import DEFAULT_GRPC_PORT, GRPC_MAX_MESSAGE_LENGTH, NOS_PROFILING_ENABLED
from nos.logging import logger
from nos.protoc import import_module
from nos.version import __version__


nos_service_pb2 = import_module("nos_service_pb2")
nos_service_pb2_grpc = import_module("nos_service_pb2_grpc")

MB_BYTES = 1024**2


@dataclass
class ClientState:
    """State of the client for serialization purposes."""

    address: str
    """Address for the gRPC server."""


class Client:
    """Main gRPC client for NOS inference service.

    Parameters:
        address (str): Address for the gRPC server.

    Usage:
        ```py

        >>> client = Client(address="localhost:50051")  # create client
        >>> client.WaitForServer()  # wait for server to start
        >>> client.CheckCompatibility()  # check compatibility with server

        >>> client.ListModels()  # list all models registered

        >>> img = Image.open("test.jpg")
        >>> visual_model = client.Module("openai/clip-vit-base-patch32")  # instantiate CLIP module
        >>> visual_model(images=img)  # predict with CLIP

        >>> fastrcnn_model = client.Module("torchvision/fasterrcnn-mobilenet-v3-large-320-fpn")  # instantiate FasterRCNN module
        >>> fastrcnn_model(images=img)
        ```
    """

    def __init__(self, address: str = f"[::]:{DEFAULT_GRPC_PORT}"):
        """Initializes the gRPC client.

        Args:
            address (str): Address for the gRPC server. Defaults to f"[::]:{DEFAULT_GRPC_PORT}".
        """
        self.address: str = address
        self._channel: grpc.Channel = None
        self._stub: nos_service_pb2_grpc.InferenceServiceStub = None
        self._uuid: str = secrets.token_hex(4)

    def __repr__(self) -> str:
        """Returns the string representation of the client.

        Returns:
            str: String representation of the client.
        """
        return f"Client(address={self.address})"

    def __getstate__(self) -> ClientState:
        """Returns the state of the client for serialization purposes.

        Returns:
            ClientState: State of the client.
        """
        return ClientState(address=self.address)

    def __setstate__(self, state: ClientState) -> None:
        """Sets the state of the client for de-serialization purposes.

        Args:
            state (ClientState): State of the client.
        Returns:
            None (NoneType): Nothing.
        """
        self.address = state.address
        self._channel = None
        self._stub = None

    @property
    def stub(self) -> nos_service_pb2_grpc.InferenceServiceStub:
        """Returns the gRPC stub.

        Note: The stub is created on-demand for serialization purposes,
        as we don't want to create a channel until we actually need it.
        This is especially useful for pickling/un-pickling the client.

        Returns:
            nos_service_pb2_grpc.InferenceServiceStub: gRPC stub.
        Raises:
            NosClientException: If the server fails to respond to the connection request.
        """
        if not self._stub:
            options = [
                ("grpc.max_message_length", GRPC_MAX_MESSAGE_LENGTH),
                ("grpc.max_send_message_length", GRPC_MAX_MESSAGE_LENGTH),
                ("grpc.max_receive_message_length", GRPC_MAX_MESSAGE_LENGTH),
            ]
            self._channel = grpc.insecure_channel(self.address, options)
            try:
                self._stub = nos_service_pb2_grpc.InferenceServiceStub(self._channel)
            except Exception as e:
                raise ServerReadyException(f"Failed to connect to server ({e})", e)
        assert self._channel
        assert self._stub
        return self._stub

    def IsHealthy(self) -> bool:
        """Check if the gRPC server is healthy.

        Returns:
            bool: True if the server is running, False otherwise.
        Raises:
            NosClientException: If the server fails to respond to the ping.
        """
        try:
            response: nos_service_pb2.PingResponse = self.stub.Ping(empty_pb2.Empty())
            return response.status == "ok"
        except grpc.RpcError as e:
            raise ServerReadyException(f"Failed to ping server (details={e.details()})", e)

    def WaitForServer(self, timeout: int = 60, retry_interval: int = 5) -> None:
        """Ping the gRPC server for health.

        Args:
            timeout (int, optional): Timeout in seconds. Defaults to 60.
            retry_interval (int, optional): Retry interval in seconds. Defaults to 5.
        Returns:
            bool: True if the server is running, False otherwise.
        Raises:
            NosClientException: If the server fails to respond to the ping or times out.
        """
        st = time.time()
        while time.time() - st <= timeout:
            try:
                return self.IsHealthy()
            except Exception:
                elapsed = time.time() - st
                if int(elapsed) > 10:
                    logger.warning("Waiting for server to start... (elapsed={:.0f}s)".format(time.time() - st))
                time.sleep(retry_interval)
        raise ServerReadyException("Failed to ping server.")

    def GetServiceVersion(self) -> str:
        """Get service version.

        Returns:
            str: Service version (e.g. 0.0.4).
        Raises:
            NosClientException: If the server fails to respond to the request.
        """
        try:
            response: nos_service_pb2.ServiceInfoResponse = self.stub.GetServiceInfo(empty_pb2.Empty())
            return response.version
        except grpc.RpcError as e:
            raise ServerReadyException(f"Failed to get service info (details={e.details()})", e)

    def GetServiceRuntime(self) -> str:
        """Get service runtime.

        Returns:
            str: Service runtime (e.g. cpu, gpu, local).
        Raises:
            NosClientException: If the server fails to respond to the request.
        """
        try:
            response: nos_service_pb2.ServiceInfoResponse = self.stub.GetServiceInfo(empty_pb2.Empty())
            return response.runtime
        except grpc.RpcError as e:
            raise ServerReadyException(f"Failed to get service info (details={e.details()})", e)

    def CheckCompatibility(self) -> bool:
        """Check if the service version is compatible with the client.

        Returns:
            bool: True if the service version is compatible, False otherwise.
        Raises:
            NosClientException: If the server fails to respond to the request.
        """
        # TODO (spillai): For now, we enforce strict version matching
        # until we have tests for client-server compatibility.
        is_compatible = self.GetServiceVersion() == __version__
        if not is_compatible:
            raise ClientException(
                f"Client-Server version mismatch (client={__version__}, server={self.GetServiceVersion()})"
            )
        return is_compatible

    def ListModels(self) -> List[ModelSpec]:
        """List all models.

        Returns:
            List[ModelSpec]: List of ModelInfo (name, task).
        Raises:
            NosClientException: If the server fails to respond to the request.
        """
        try:
            response: nos_service_pb2.GenericResponse = self.stub.ListModels(empty_pb2.Empty())
            models: List[str] = loads(response.response_bytes)
            return list(models)
        except grpc.RpcError as e:
            raise ClientException(f"Failed to list models (details={e.details()})", e)

    def LoadModel(self, model_id: str, num_replicas: int = 1) -> None:
        """Load a model.

        Args:
            model_id (str): Name of the model to load.
            num_replicas (int, optional): Number of replicas to load. Defaults to 1.
        Raises:
            NosClientException: If the server fails to respond to the request.
        """
        try:
            self.stub.LoadModel(
                nos_service_pb2.GenericRequest(request_bytes=dumps({"id": model_id, "num_replicas": num_replicas}))
            )
        except grpc.RpcError as e:
            raise ClientException(f"Failed to load model (details={e.details()})", e)

    @lru_cache()  # noqa: B019
    def _get_model_catalog(self) -> ModelSpecMetadataCatalog:
        """Get the model catalog and cache.

        Returns:
            Dict[str, ModelSpec]: Model catalog (name, task).
        Raises:
            NosClientException: If the server fails to respond to the request.
        """
        try:
            response: nos_service_pb2.GenericResponse = self.stub.GetModelCatalog(empty_pb2.Empty())
            ModelSpecMetadataCatalog._instance = loads(response.response_bytes)
            return ModelSpecMetadataCatalog.get()
        except grpc.RpcError as e:
            raise ClientException(f"Failed to get model catalog (details={e.details()})", e)

    def GetModelInfo(self, model_id: str) -> ModelSpec:
        """Get the relevant model information from the model name.

        Note: This may be possible only after initialization, as we need to inspect the
        HW to understand the configurable image resolutions, batch sizes etc.

        Args:
            spec (ModelSpec): Model information.
        """
        try:
            # Update the model catalog so that the metadata is cached on the client-side
            _ = self._get_model_catalog()
            # Get the model spec separately
            response: nos_service_pb2.GenericResponse = self.stub.GetModelInfo(
                wrappers_pb2.StringValue(value=model_id)
            )
            model_spec: ModelSpec = loads(response.response_bytes)
            return model_spec
        except grpc.RpcError as e:
            raise ClientException(f"Failed to get model info (details={(e.details())})", e)

    @lru_cache(maxsize=8)  # noqa: B019
    def Module(self, model_id: str, shm: bool = False) -> "Module":
        """Instantiate a model module.

        Args:
            model_id (str): Name of the model to init.
            shm (bool, optional): Enable shared memory transport. Defaults to False.
        Returns:
            Module: Inference module.
        """
        return Module(model_id, self, shm=shm)

    @lru_cache(maxsize=8)  # noqa: B019
    def ModuleFromSpec(self, spec: ModelSpec, shm: bool = False) -> "Module":
        """Instantiate a model module from a model spec.

        Args:
            spec (ModelSpec): Model specification.
            shm (bool, optional): Enable shared memory transport. Defaults to False.
        Returns:
            Module: Inference module.
        """
        return Module(spec.task, spec.name, self, shm=shm)

    def ModuleFromCls(self, cls: Callable, shm: bool = False) -> "Module":
        raise NotImplementedError("ModuleFromCls not implemented yet.")

    def _upload_file(self, path: Path, chunk_size: int = 4 * MB_BYTES) -> Path:
        """Upload a file to the server.

        Args:
            path (Path): Path to the file to be uploaded.
        Returns:
            path: Temporary remote path of the uploaded file.
        Raises:
            NosClientException: If the server fails to respond to the request.
        """
        try:
            response = None
            with path.open("rb") as f:
                for cidx, chunk in enumerate(iter(lambda: f.read(chunk_size), b"")):
                    response: nos_service_pb2.GenericResponse = self.stub.UploadFile(
                        iter(
                            [
                                nos_service_pb2.GenericRequest(
                                    request_bytes=dumps(
                                        {"chunk_bytes": chunk, "chunk_index": cidx, "filename": str(path)}
                                    )
                                )
                            ]
                        )
                    )
            return Path(loads(response.response_bytes)["filename"])
        except grpc.RpcError as e:
            raise ClientException(f"Failed to upload file (details={e.details()})", e)

    def _delete_file(self, path: Path) -> None:
        """Delete a file from the server.

        Args:
            path (Path): Path to the file to be deleted.
        Raises:
            NosClientException: If the server fails to respond to the request.
        """
        try:
            self.stub.DeleteFile(nos_service_pb2.GenericRequest(request_bytes=dumps({"filename": str(path)})))
        except grpc.RpcError as e:
            raise ClientException(f"Failed to delete file (details={e.details()})", e)

    @contextlib.contextmanager
    def UploadFile(self, path: Path, chunk_size: int = 4 * MB_BYTES) -> Path:
        """Upload a file to the server, and delete it after use."""
        if not path.exists():
            raise FileNotFoundError(f"File not found [path={path}]")
        try:
            logger.debug(f"Uploading file [path={path}]")
            remote_path: Path = self._upload_file(path, chunk_size=chunk_size)
            logger.debug(f"Uploaded file [path={path}, remote_path={remote_path}]")
            yield remote_path
            logger.debug(f"Deleting file [path={path}, remote_path={remote_path}]")
        except Exception as e:
            logger.error(f"Failed to upload file [path={path}, e={e}]")
        finally:
            logger.debug(f"Deleting file [path={path}, remote_path={remote_path}]")
            try:
                self._delete_file(path)
            except Exception as e:
                logger.error(f"Failed to delete file [path={path}, remote_path={remote_path}, e={e}]")
            logger.debug(f"Deleted file [path={path}, remote_path={remote_path}]")

    def Run(
        self,
        model_id: str,
        inputs: Dict[str, Any],
        method: str = None,
        shm: bool = False,
    ) -> nos_service_pb2.GenericResponse:
        """Run module.

        Args:
            model_id (str):
                Model identifier (e.g. openai/clip-vit-base-patch32).
            inputs (Dict[str, Any]): Inputs to the model ("images", "texts", "prompts" etc) as
                defined in the ModelSpec.signature.
            method (str, optional): Method to call on the model. Defaults to None.
            stream (bool, optional): Stream the response. Defaults to False.
            shm (bool, optional): Enable shared memory transport. Defaults to False.
        Returns:
            nos_service_pb2.GenericResponse: Inference response.
        Raises:
            NosClientException: If the server fails to respond to the request.
        """
        module: Module = self.Module(model_id, shm=shm)
        return module(**inputs, _method=method)

    def Stream(
        self,
        model_id: str,
        inputs: Dict[str, Any],
        method: str = None,
        shm: bool = False,
    ) -> Iterable[nos_service_pb2.GenericResponse]:
        """Run module in streaming mode."""
        assert shm is False, "Shared memory transport is not supported for streaming yet."
        module: Module = self.Module(model_id, shm=shm)
        return module(**inputs, _method=method, _stream=True)


@dataclass
class Module:
    """Inference module for remote model execution.

    Usage:
        ```python
        # Create client
        >>> client = Client()
        # Instantiate new task module with specific model name
        >>> model = client.Module("openai/clip-vit-base-patch32")
        # Predict with model using `__call__`
        >>> predictions = model({"images": img})
        ```
    """

    id: str
    """Model identifier (e.g. openai/clip-vit-base-patch32)."""
    _client: Client
    """gRPC client."""
    shm: bool = False
    """Enable shared memory transport."""
    _spec: ModelSpec = field(init=False)
    """Model specification for this module."""
    _shm_objects: Dict[str, Any] = field(init=False, default_factory=dict)
    """Shared memory data."""

    def __post_init__(self):
        """Initialize the spec."""
        self._spec = self._client.GetModelInfo(self.id)
        assert self._spec.id == self.id
        if not NOS_SHM_ENABLED or not self.shm:
            # Note (spillai): Shared memory caveats.
            # - only supported for numpy arrays
            # - registered once per module
            # - can not handle shm objects while calling multiple methods cleanly
            #   (i.e. expects the same method to be called for a module)
            self._shm_objects = None  # disables shm, and avoids registering/unregistering

        # Patch the module with methods from model spec signature
        for method in self._spec.signature.keys():
            if hasattr(self, method):
                # If the method to patch is __call__ just log a debug message and skip,
                # otherwise log a warning so that the user is warned that the method is skipped.
                log = logger.debug if method == "__call__" else logger.warning
                log(f"Module ({self.id}) already has method ({method}), skipping ...")
                continue

            assert self._spec.signature[method].method == method
            # Patch the module with the partial method only if the default method is
            # not the same as the method being patched i.e., there's no need to pass
            # `method`` to the partial method since it's already the default method.
            if self._spec.default_method != method:
                method_partial = partial(self.__call__, _method=method)
            else:
                method_partial = self.__call__
            setattr(self, method, method_partial)
            logger.debug(f"Module ({self.id}) patched [method={method}].")
        logger.debug(f"Module ({self.id}) initialized [spec={self._spec}, shm={self.shm}].")

    @property
    def stub(self) -> nos_service_pb2_grpc.InferenceServiceStub:
        return self._client.stub

    @property
    def client_id(self) -> str:
        """Correlation ID for this module."""
        return self._client._uuid

    @cached_property
    def object_id(self) -> str:
        """Unique object ID for this module."""
        return f"{self._spec.id}_{secrets.token_hex(4)}"

    @cached_property
    def namespace(self) -> str:
        """Unique namespace for this module."""
        return f"{self.client_id}/{self.object_id}"

    def _encode(self, inputs: Dict[str, Any], method: str = None) -> Dict[str, Any]:
        """Encode the inputs dictionary for transmission.
        TODO (spillai)
            - Support middlewares for encoding/decoding.
            - Validate inputs/outputs with spec signature.
            - Support shared memory transport.
            - SerDe before/after transmission.
        Args:
            inputs (Dict[str, Any]): Inputs to the model ("images", "texts", "prompts" etc) as
                defined in the ModelSpec.signature.
        Returns:
            Dict[str, Any]: Encoded inputs.
        """
        # Validate data with spec signature
        if method is None:
            method = self._spec.default_method
        if method not in self._spec.signature:
            raise InferenceException(f"Method {method} not found in spec signature.")
        sig: FunctionSignature = self._spec.signature[method]
        inputs = FunctionSignature.validate(inputs, sig.parameters)

        # Encode List[np.ndarray] as stacked np.ndarray (B, H, W, C)
        for k, v in inputs.items():
            if isinstance(v, Image.Image):
                inputs[k] = np.asarray(v)
            elif isinstance(v, list) and isinstance(v[0], Image.Image):
                inputs[k] = np.stack([np.asarray(_v) for _v in v], axis=0)
            elif isinstance(v, list) and isinstance(v[0], np.ndarray):
                inputs[k] = np.stack(v, axis=0)

        # Optionally, create/register shm and copy over numpy arrays to shm
        if self._shm_objects is not None:
            # If inputs are already registered, check if they've changed
            # If they've changed, unregister and re-register.
            # Checks: 1) keys match, 2) inputs are np.ndarray, 3) shapes match
            if len(self._shm_objects):
                valid = inputs.keys() == self._shm_objects.keys()
                for k, v in inputs.items():
                    try:
                        valid &= isinstance(v, np.ndarray)
                        if valid and isinstance(v, np.ndarray):
                            valid &= v.shape == self._shm_objects[k].shape
                    except Exception:
                        valid = False
                if not valid:
                    logger.debug(
                        """Inputs are inconsistent with previously registered shared memory objects, unregistering ..."""
                    )
                    registered_str = [(k, type(v), v.shape) for k, v in self._shm_objects.items()]
                    inputs_str = [
                        (k, type(v), v.shape if isinstance(v, np.ndarray) else None) for k, v in inputs.items()
                    ]
                    logger.debug(
                        f"""Unregistering due to inconsistent shapes ... [registered={registered_str}, """
                        f"""inputs={inputs_str}]"""
                    )
                    self.UnregisterSystemSharedMemory()

            # Register system shared memory for inputs, if not already registered
            if not len(self._shm_objects):
                self.RegisterSystemSharedMemory(inputs)

        # Copy data from numpy array to shared memory
        if self._shm_objects is not None and len(self._shm_objects):
            inputs = SharedMemoryTransportManager.copy(self._shm_objects, inputs)

        # Pickle the data for transmission
        return {k: dumps(v) for k, v in inputs.items()}

    def _decode(self, response_bytes: bytes) -> Any:
        """Decode the response bytes."""
        return loads(response_bytes)

    def __del__(self):
        """Delete the shared memory."""
        if self._shm_objects is not None:
            self.UnregisterSystemSharedMemory()

    def GetModelInfo(self) -> ModelSpec:
        """Get the relevant model information from the model name."""
        return self._spec

    def Load(self, num_replicas: int = 1) -> None:
        """Load the model."""
        return self._client.LoadModel(self.id, num_replicas=num_replicas)

    def RegisterSystemSharedMemory(self, inputs: Dict[str, Any]) -> None:
        """Register system shared memory for inputs.

        Args:
            inputs (Dict[str, Any]): Inputs to the model ("images", "texts", "prompts" etc) as
                defined in the ModelSpec.signature. For example, {"images": np.ndarray}.
        """
        # Create shared memory request
        # We convert the numpy arrays to TensorSpec(s) to let the
        # server know the shape and dtype of the underlying shm data.
        if not NOS_SHM_ENABLED:
            logger.warning("Shared memory is not enabled, skipping.")
            return

        shm_request = {}
        for k, v in inputs.items():
            if isinstance(v, np.ndarray):
                shm_request[k] = TensorSpec(v.shape, dtype=str(v.dtype))
        if not len(shm_request):
            logger.debug(f"Skipping shm registration, no numpy arrays found in inputs [inputs={inputs}]")
            return
        logger.debug(f"Registering shm [request={shm_request}]")

        # Request shared memory, fail gracefully if not supported
        try:
            # Clear the cached object_id and namespace so that they are re-initialized
            if "object_id" in self.__dict__:  # noqa: WPS421
                del self.object_id
                del self.namespace
            response = self.stub.RegisterSystemSharedMemory(
                nos_service_pb2.GenericRequest(request_bytes=dumps(shm_request)),
                metadata=[("client_id", self.client_id), ("object_id", self.object_id)],
            )

            # Register the shared memory objects by name on the client
            # Note (spillai): This calls __setstate__ on the SharedMemoryNumpyObject
            self._shm_objects = loads(response.response_bytes)
            logger.debug(f"Registered shm [namespace={self.namespace}, objects={self._shm_objects}]")
        except grpc.RpcError as e:
            logger.debug(f"Failed to register shm [request={shm_request}, e={e.details()}], skipping.")
            self._shm_objects = None

    def UnregisterSystemSharedMemory(self) -> None:
        """Unregister system shared memory."""
        if self._shm_objects is None:
            logger.warning("Shared memory is not enabled, skipping.")
            return

        if len(self._shm_objects):
            logger.debug(
                f"Unregistering shm [namespace={self.namespace}, objects={[(k, v) for k, v in self._shm_objects.items()]}"
            )

            # Close the shared memory objects
            shm_objects_name_map = {k: v.name for k, v in self._shm_objects.items()}
            for _k, v in self._shm_objects.items():
                v.close()

            # Unregister the shared memory objects on the server
            try:
                self.stub.UnregisterSystemSharedMemory(
                    nos_service_pb2.GenericRequest(request_bytes=dumps(shm_objects_name_map)),
                    metadata=[("client_id", self.client_id), ("object_id", self.object_id)],
                )
                # Delete the shared memory objects after safely closing (client-side) and unregistering them (server-side).
                self._shm_objects = {}
                logger.debug(f"Unregistered shm [{self._shm_objects}]")
            except grpc.RpcError as e:
                logger.error(f"Failed to unregister shm [{self._shm_objects}], error: {e.details()}")
                raise ClientException(f"Failed to unregister shm [{self._shm_objects}]", e)

    def __call__(self, _method: str = None, _stream: bool = False, **inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Call the instantiated module/model.

        Args:
            _method (str, optional): Method to call on the model. Defaults to None.
            _stream (bool, optional): Stream the response. Defaults to False.
            **inputs (Dict[str, Any]): Inputs to the model ("images", "texts", "prompts" etc) as
                defined in ModelSpec.signature.
        Returns:
            Dict[str, Any]: Inference response.
        Raises:
            NosInputValidationException: If the inputs are inconsistent with the spec signature.
            NosInferenceException: If the server fails to respond to the request.
            NosClientException: If the outputs cannot be decoded.

        Note: While encoding the inputs, we check if the input dictionary is consistent
        with inputs/outputs defined in `spec.signature` and only then encode it.
        """
        # Encode the inputs
        st = time.perf_counter()
        try:
            inputs = self._encode(inputs, method=_method)
        except Exception as e:
            logger.error(f"Failed to encode inputs [model={self.id}, method={_method}, inputs={inputs}, e={e}]")
            raise InputValidationException(
                f"Failed to encode inputs [model={self.id}, method={_method}, inputs={inputs}, e={e}]", e
            )
        if NOS_PROFILING_ENABLED:
            logger.debug(f"Encoded inputs [model={self.id}, elapsed={(time.perf_counter() - st) * 1e3:.1f}ms]")

        # Prepare the request
        request = nos_service_pb2.GenericRequest(
            request_bytes=dumps(
                {
                    "id": self._spec.id,
                    "method": _method,
                    "inputs": inputs,
                }
            )
        )
        try:
            # Execute the request
            st = time.perf_counter()
            logger.debug(f"Executing request [model={self.id}]")
            if not _stream:
                response: nos_service_pb2.GenericResponse = self.stub.Run(request)
            else:
                response: Iterable[nos_service_pb2.GenericResponse] = self.stub.Stream(request)
            if NOS_PROFILING_ENABLED:
                logger.debug(f"Executed request [model={self.id}, elapsed={(time.perf_counter() - st) * 1e3:.1f}ms]")
        except grpc.RpcError as e:
            logger.error(f"Run() failed [details={e.details()}, inputs={inputs.keys()}]")
            raise InferenceException(f"Run() failed [model={self.id}, details={e.details()}]", e)

        # Decode / stream the response
        st = time.perf_counter()
        try:
            if not _stream:
                response = self._decode(response.response_bytes)
            else:
                return _StreamingModuleResponse(response, self._decode)
        except Exception as e:
            logger.error(f"Failed to decode response [model={self.id}, e={e}]")
            raise ClientException(f"Failed to decode response [model={self.id}, e={e}]", e)
        if NOS_PROFILING_ENABLED:
            logger.debug(f"Decoded response [model={self.id}, elapsed={(time.perf_counter() - st) * 1e3:.1f}ms]")
        return response


@dataclass
class _StreamingModuleResponse:
    response: Iterable[nos_service_pb2.GenericResponse]
    fn: Callable

    def __iter__(self) -> Any:
        return self

    def __next__(self) -> Any:
        resp = next(self.response)
        return self.fn(resp.response_bytes)
