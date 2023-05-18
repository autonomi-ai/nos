"""gRPC client for NOS service."""
from contextlib import contextmanager
from dataclasses import dataclass
from typing import List, Union

import cloudpickle
import grpc
import numpy as np
from google.protobuf import empty_pb2
from PIL import Image

from nos.client.exceptions import NosClientException
from nos.constants import DEFAULT_GRPC_PORT
from nos.logging import logger
from nos.protoc import import_module


nos_service_pb2 = import_module("nos_service_pb2")
nos_service_pb2_grpc = import_module("nos_service_pb2_grpc")


@contextmanager
def InferenceSession(stub: nos_service_pb2_grpc.InferenceServiceStub, model_name: str, num_replicas: int = 1):
    """Remote model context manager.

    Args:
        stub (nos_service_pb2_grpc.InferenceServiceStub): gRPC stub.
        model_name (str): Name of the model to init.
        num_replicas (int): Number of replicas to init.

    Yields:
        None (NoneType): Nothing.
    """
    # Create inference stub and init model
    request = nos_service_pb2.InitModelRequest(model_name=model_name, num_replicas=num_replicas)
    response: nos_service_pb2.InitModelResponse = stub.InitModel(request)
    logger.info(f"Init Model response: {response}")
    # Yield so that the model inference can be done
    yield
    # Delete model
    request = nos_service_pb2.DeleteModelRequest(model_name=model_name)
    response: nos_service_pb2.DeleteModelResponse = stub.DeleteModel(request)
    logger.info(f"Delete Model response: {response}")


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
        # Create client
        >>> client = InferenceClient(address="localhost:50051")

        # List all models registered with the server
        # models = ['openai/clip-vit-base-patch32', ...]
        >>> models = models client.ListModels()

        # Encode "Hello world!" with the CLIP text encoder
        >>> client.Predict(method=MethodType.TXT2VEC, model_name="openai/clip-vit-base-patch32", text="Hello world!")

        # Encode img with the CLIP visual encoder
        >>> client.Predict(method=MethodType.IMG2VEC, model_name="openai/clip-vit-base-patch32", text="Hello world!")

        # Predict bounding-boxes from with FastRCNN
        >>> img = Image.open("test.jpg")
        >>> client.Predict(method=MethodType.IMG2BBOX, model_name="torchvision/fasterrcnn_mobilenet_v3_large_320_fpn", img=img)
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
        """
        if not self._stub:
            self._channel = grpc.insecure_channel(self.address)
            self._stub = nos_service_pb2_grpc.InferenceServiceStub(self._channel)
        assert self._channel
        assert self._stub
        return self._stub

    def ListModels(self) -> List[str]:
        """List all models.

        Returns:
            List[str]: List of model names.
        """
        try:
            response: nos_service_pb2.ModelListResponse = self.stub.ListModels(empty_pb2.Empty())
            logger.debug(response.models)
            return list(response.models)
        except grpc.RpcError as e:
            logger.error(f"Failed to list models ({e})")

    def GetModelInfo(self, model_name: str):
        """Get the relevant model information from the model name.

        Note: This may be possible only after initialization, as we need to inspect the
        HW to understand the configurable image resolutions, batch sizes etc.

        Args:
            model_name (str): Model identifier (e.g. openai/clip-vit-base-patch32).
        """
        raise NotImplementedError()

    def Predict(
        self,
        method: str,
        model_name: str,
        img: Union[Image.Image, np.ndarray, List[Image.Image], List[np.ndarray]] = None,
        text: str = None,
    ) -> nos_service_pb2.InferenceResponse:
        """Predict with model identifier.

        Args:
            method (str):
                Method to use for prediction (one of MethodType.TXT2VEC, MethodType.IMG2VEC,
                MethodType.IMG2BBOX, MethodType.TXT2IMG).
            model_name (str):
                Model identifier (e.g. openai/clip-vit-base-patch32).
            img (Union[Image.Image, np.ndarray, List[Image.Image], List[np.ndarray]]):
                Image or text to predict on.
            text (str): Prompt text to use for text-generation or embedding.

        Returns:
            nos_service_pb2.InferenceResponse: Inference response.
        """
        if method not in ("txt2vec", "img2vec", "img2bbox", "txt2img"):
            raise NosClientException(f"Invalid method {method}")

        try:
            if method in ("txt2vec", "txt2img"):
                response = self.stub.Predict(
                    nos_service_pb2.InferenceRequest(
                        method=method,
                        model_name=model_name,
                        text_request=nos_service_pb2.TextRequest(text=text),
                    )
                )
            elif method in ("img2vec", "img2bbox"):
                response = self.stub.Predict(
                    nos_service_pb2.InferenceRequest(
                        method=method,
                        model_name=model_name,
                        image_request=nos_service_pb2.ImageRequest(image_bytes=cloudpickle.dumps(img, protocol=4)),
                    )
                )
            response = cloudpickle.loads(response.result)
            logger.debug(response)
            return response
        except grpc.RpcError as e:
            logger.error(f"Failed to predict with model {model_name} ({e})")
            raise NosClientException(f"Failed to predict with model {model_name} ({e})")
