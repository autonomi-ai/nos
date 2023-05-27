"""gRPC client for NOS service."""
import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Callable, Dict, List

import grpc
from google.protobuf import empty_pb2

from nos.client.exceptions import NosClientException
from nos.common import ModelSpec, TaskType, loads
from nos.constants import DEFAULT_GRPC_PORT
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
                raise NosClientException(f"Failed to connect to server ({e})")
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
        except grpc.RpcError as exc:
            raise NosClientException(f"Failed to ping server ({exc})")

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
        exc = None
        st = time.time()
        while time.time() - st <= timeout:
            try:
                return self.IsHealthy()
            except Exception:
                logger.warning("Waiting for server to start... (elapsed={:.0f}s)".format(time.time() - st))
                time.sleep(retry_interval)
        raise NosClientException(f"Failed to ping server ({exc})")

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
            raise NosClientException(f"Failed to get service info ({e})")

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
            List[ModelInfo]: List of ModelInfo (name, task).
        Raises:
            NosClientException: If the server fails to respond to the request.
        """
        try:
            response: nos_service_pb2.ModelListResponse = self.stub.ListModels(empty_pb2.Empty())
            logger.debug(response.models)
            return [ModelSpec(name=minfo.name, task=TaskType(minfo.task)) for minfo in response.models]
        except grpc.RpcError as e:
            raise NosClientException(f"Failed to list models ({e})")

    def GetModelInfo(self, model_spec: ModelSpec) -> ModelSpec:
        """Get the relevant model information from the model name.

        Note: This may be possible only after initialization, as we need to inspect the
        HW to understand the configurable image resolutions, batch sizes etc.

        Args:
            model_spec (ModelSpec): Model specification.
        """
        try:
            response: nos_service_pb2.ModelInfoResponse = self.stub.GetModelInfo(
                nos_service_pb2.ModelInfoRequest(
                    request=nos_service_pb2.ModelInfo(task=model_spec.task.value, name=model_spec.name)
                )
            )
            logger.debug(response)
            spec: ModelSpec = loads(response.response_bytes)
            return spec
        except grpc.RpcError as e:
            raise NosClientException(f"Failed to get model info ({e})")

    @lru_cache(maxsize=32)  # noqa: B019
    def Module(self, task: TaskType, model_name: str) -> "InferenceModule":
        """Instantiate a model module.

        Args:
            task (TaskType): Task used for prediction.
            model_name (str): Name of the model to init.
        Returns:
            InferenceModule: Inference module.
        """
        return InferenceModule(task, model_name, self)

    @lru_cache(maxsize=32)  # noqa: B019
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

    def __post_init__(self):
        """Initialize the spec."""
        self._spec = self._client.GetModelInfo(ModelSpec(name=self.model_name, task=self.task))

    @property
    def stub(self):
        return self._client.stub

    def GetModelInfo(self) -> ModelSpec:
        """Get the relevant model information from the model name."""
        return self._spec

    def __call__(self, **inputs: Dict[str, Any]) -> nos_service_pb2.InferenceResponse:
        """Call the instantiated module/model.

        Args:
            **inputs (Dict[str, Any]): Inputs to the model ("images", "texts", "prompts" etc) as
                defined in the ModelSpec.signature.inputs.
        Returns:
            nos_service_pb2.InferenceResponse: Inference response.
        Raises:
            NosClientException: If the server fails to respond to the request.
        """
        # Check if the input dictionary is consistent
        # with inputs/outputs defined in `spec.signature`
        # and then encode it.
        inputs = self._spec.signature.encode_inputs(inputs)
        request = nos_service_pb2.InferenceRequest(
            model=nos_service_pb2.ModelInfo(
                task=self.task.value,
                name=self.model_name,
            ),
            inputs=inputs,
        )
        try:
            response = self.stub.Run(request)
            response = loads(response.response_bytes)
            logger.debug(response)
            return response
        except grpc.RpcError as e:
            raise NosClientException(f"Failed to run model {self.model_name} ({e})")
