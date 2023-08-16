"""gRPC client for NOS service."""
import secrets
import time
from dataclasses import dataclass, field
from functools import cached_property, lru_cache
from typing import Any, Callable, Dict, List

import grpc
import numpy as np
from google.protobuf import empty_pb2
from PIL import Image

from nos.common import FunctionSignature, ModelSpec, TaskType, TensorSpec, dumps, loads
from nos.common.exceptions import (
    NosClientException,
    NosInferenceException,
    NosInputValidationException,
    NosServerReadyException,
)
from nos.common.shm import NOS_SHM_ENABLED, SharedMemoryTransportManager
from nos.constants import DEFAULT_GRPC_PORT, NOS_PROFILING_ENABLED
from nos.logging import logger
from nos.protoc import import_module
from nos.version import __version__


nos_service_pb2 = import_module("nos_service_pb2")
nos_service_pb2_grpc = import_module("nos_service_pb2_grpc")


@dataclass
class InferenceClientState:
    """State of the client for serialization purposes."""

    address: str
    """Address for the gRPC server."""


class InferenceClient:
    """Main gRPC client for NOS inference service.

    Parameters:
        address (str): Address for the gRPC server.

    Usage:
        ```py

        >>> client = InferenceClient(address="localhost:50051")  # create client
        >>> client.WaitForServer()  # wait for server to start
        >>> client.CheckCompatibility()  # check compatibility with server

        >>> client.ListModels()  # list all models registered

        >>> text_model = client.Module(TaskType.TEXT_EMBEDDING, "openai/clip-vit-base-patch32")  # instantiate CLIP module
        >>> text_model(text="Hello world!")  # predict with CLIP

        >>> img = Image.open("test.jpg")
        >>> visual_model = client.Module(TaskType.IMAGE_EMBEDDING, "openai/clip-vit-base-patch32")  # instantiate CLIP module
        >>> visual_model(images=img)  # predict with CLIP

        >>> fastrcnn_model = client.Module(TaskType.OBJECT_DETECTION_2D, "torchvision/fasterrcnn_mobilenet_v3_large_320_fpn")  # instantiate FasterRCNN module
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

    def __getstate__(self) -> InferenceClientState:
        """Returns the state of the client for serialization purposes.

        Returns:
            InferenceClientState: State of the client.
        """
        return InferenceClientState(address=self.address)

    def __setstate__(self, state: InferenceClientState) -> None:
        """Sets the state of the client for de-serialization purposes.

        Args:
            state (InferenceClientState): State of the client.
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
            self._channel = grpc.insecure_channel(self.address)
            try:
                self._stub = nos_service_pb2_grpc.InferenceServiceStub(self._channel)
            except Exception as e:
                raise NosServerReadyException(f"Failed to connect to server ({e})", e)
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
            raise NosServerReadyException(f"Failed to ping server (details={e.details()})", e)

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
                logger.warning("Waiting for server to start... (elapsed={:.0f}s)".format(time.time() - st))
                time.sleep(retry_interval)
        raise NosServerReadyException("Failed to ping server.")

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
            raise NosServerReadyException(f"Failed to get service info (details={e.details()})", e)

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
            raise NosClientException(
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
            response: nos_service_pb2.ModelListResponse = self.stub.ListModels(empty_pb2.Empty())
            return [ModelSpec(name=minfo.name, task=TaskType(minfo.task)) for minfo in response.models]
        except grpc.RpcError as e:
            raise NosClientException(f"Failed to list models (details={e.details()})", e)

    def GetModelInfo(self, spec: ModelSpec) -> ModelSpec:
        """Get the relevant model information from the model name.

        Note: This may be possible only after initialization, as we need to inspect the
        HW to understand the configurable image resolutions, batch sizes etc.

        Args:
            spec (ModelSpec): Model information.
        """
        try:
            response: nos_service_pb2.ModelInfoResponse = self.stub.GetModelInfo(
                nos_service_pb2.ModelInfoRequest(
                    request=nos_service_pb2.ModelInfo(task=spec.task.value, name=spec.name)
                )
            )
            model_spec: ModelSpec = loads(response.response_bytes)
            return model_spec
        except grpc.RpcError as e:
            raise NosClientException(f"Failed to get model info (details={(e.details())})", e)

    @lru_cache(maxsize=8)  # noqa: B019
    def Module(self, task: TaskType, model_name: str) -> "InferenceModule":
        """Instantiate a model module.

        Args:
            task (TaskType): Task used for prediction.
            model_name (str): Name of the model to init.
        Returns:
            InferenceModule: Inference module.
        """
        return InferenceModule(task, model_name, self)

    @lru_cache(maxsize=8)  # noqa: B019
    def ModuleFromSpec(self, spec: ModelSpec) -> "InferenceModule":
        """Instantiate a model module from a model spec.

        Args:
            spec (ModelSpec): Model specification.
        Returns:
            InferenceModule: Inference module.
        """
        return InferenceModule(spec.task, spec.name, self)

    def ModuleFromCls(self, cls: Callable) -> "InferenceModule":
        raise NotImplementedError("ModuleFromCls not implemented yet.")

    def Run(
        self,
        task: TaskType,
        model_name: str,
        **inputs: Dict[str, Any],
    ) -> nos_service_pb2.InferenceResponse:
        """Run module.

        Args:
            task (TaskType): Task used for prediction.
                Tasks supported:
                    (TaskType.OBJECT_DETECTION_2D, TaskType.IMAGE_SEGMENTATION_2D,
                    TaskType.IMAGE_CLASSIFICATION, TaskType.IMAGE_GENERATION,
                    TaskType.IMAGE_EMBEDDING, TaskType.TEXT_EMBEDDING)
            model_name (str):
                Model identifier (e.g. openai/clip-vit-base-patch32).
            **inputs (Dict[str, Any]): Inputs to the model ("images", "texts", "prompts" etc) as
                defined in the ModelSpec.signature.inputs.
        Returns:
            nos_service_pb2.InferenceResponse: Inference response.
        Raises:
            NosClientException: If the server fails to respond to the request.
        """
        module: InferenceModule = self.Module(task, model_name)
        return module(**inputs)


@dataclass
class InferenceModule:
    """Inference module for remote model execution.

    Usage:
        ```python
        # Create client
        >>> client = InferenceClient()
        # Instantiate new task module with specific model name
        >>> model = client.Module(TaskType.IMAGE_EMBEDDING, "openai/clip-vit-base-patch32")
        # Predict with model using `__call__`
        >>> predictions = model({"images": img})
        ```
    """

    task: TaskType
    """Task used for prediction.
       (TaskType.OBJECT_DETECTION_2D, TaskType.IMAGE_SEGMENTATION_2D,
        TaskType.IMAGE_CLASSIFICATION, TaskType.IMAGE_GENERATION,
        TaskType.IMAGE_EMBEDDING, TaskType.TEXT_EMBEDDING)
    """
    model_name: str
    """Model identifier (e.g. openai/clip-vit-base-patch32)."""
    _client: InferenceClient
    """gRPC client."""
    _spec: ModelSpec = field(init=False)
    """Model specification for this module."""
    _shm_objects: Dict[str, Any] = field(init=False, default_factory=dict)
    """Shared memory data."""

    def __post_init__(self):
        """Initialize the spec."""
        self._spec = self._client.GetModelInfo(ModelSpec(name=self.model_name, task=self.task))
        if not NOS_SHM_ENABLED:
            logger.debug("Shared memory disabled.")
            self._shm_objects = None  # disables shm, and avoids registering/unregistering

    @property
    def stub(self):
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

    def _encode(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Encode the inputs dictionary for transmission.
        TODO (spillai)
            - Support middlewares for encoding/decoding.
            - Validate inputs/outputs with spec signature.
            - Support shared memory transport.
            - SerDe before/after transmission.
        Args:
            inputs (Dict[str, Any]): Inputs to the model ("images", "texts", "prompts" etc) as
                defined in the ModelSpec.signature.inputs.
        Returns:
            Dict[str, Any]: Encoded inputs.
        """
        # Validate data with spec signature
        inputs = FunctionSignature.validate(inputs, self._spec.signature.inputs)

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

    def RegisterSystemSharedMemory(self, inputs: Dict[str, Any]) -> None:
        """Register system shared memory for inputs.

        Args:
            inputs (Dict[str, Any]): Inputs to the model ("images", "texts", "prompts" etc) as
                defined in the ModelSpec.signature.inputs. For example, {"images": np.ndarray}.
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
                raise NosClientException(f"Failed to unregister shm [{self._shm_objects}]", e)

    def __call__(self, **inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Call the instantiated module/model.

        Args:
            **inputs (Dict[str, Any]): Inputs to the model ("images", "texts", "prompts" etc) as
                defined in the ModelSpec.signature.inputs.
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
            inputs = self._encode(inputs)
        except Exception as e:
            logger.error(f"Failed to encode inputs [model={self.model_name}, inputs={inputs}, e={e}]")
            raise NosInputValidationException(
                f"Failed to encode inputs [model={self.model_name}, inputs={inputs}, e={e}]", e
            )
        if NOS_PROFILING_ENABLED:
            logger.debug(f"Encoded inputs [model={self._spec.name}, elapsed={(time.perf_counter() - st) * 1e3:.1f}ms]")

        try:
            # Prepare the request
            request = nos_service_pb2.InferenceRequest(
                model=nos_service_pb2.ModelInfo(
                    task=self.task.value,
                    name=self.model_name,
                ),
                inputs=inputs,
            )
            # Execute the request
            st = time.perf_counter()
            logger.debug(f"Executing request [model={self._spec.name}]]")
            response = self.stub.Run(request)
            if NOS_PROFILING_ENABLED:
                logger.debug(
                    f"Executed request [model={self._spec.name}, elapsed={(time.perf_counter() - st) * 1e3:.1f}ms]"
                )
        except grpc.RpcError as e:
            logger.error(f"Run() failed [details={e.details()}, request={request}, inputs={inputs.keys()}]")
            raise NosInferenceException(f"Run() failed [model={self.model_name}, details={e.details()}]", e)

        # Decode the response
        st = time.perf_counter()
        try:
            response = self._decode(response.response_bytes)
            response = {k: loads(v) for k, v in response.items()}
        except Exception as e:
            logger.error(f"Failed to decode response [model={self.model_name}, e={e}]")
            raise NosClientException(f"Failed to decode response [model={self.model_name}, e={e}]", e)
        if NOS_PROFILING_ENABLED:
            logger.debug(
                f"Decoded response [model={self._spec.name}, elapsed={(time.perf_counter() - st) * 1e3:.1f}ms]"
            )
        return response
