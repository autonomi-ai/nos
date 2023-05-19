from collections import OrderedDict
from dataclasses import dataclass
from typing import Union

import cloudpickle
import grpc
import numpy as np
import ray
import rich.console
import rich.status
import torch
from google.protobuf import empty_pb2
from PIL import Image

from nos import hub
from nos.constants import DEFAULT_GRPC_PORT  # noqa F401
from nos.exceptions import ModelNotFoundError
from nos.executors.ray import RayExecutor
from nos.hub import MethodType, ModelSpec
from nos.logging import logger
from nos.protoc import import_module


nos_service_pb2 = import_module("nos_service_pb2")
nos_service_pb2_grpc = import_module("nos_service_pb2_grpc")


class FixedLengthFIFODict(OrderedDict):
    """Fixed length FIFO dictionary.

    Note: Needs to be refactored to be more generic with FIFO/LRU options.
    """

    def __init__(self, *args, **kwargs):
        self._maxlen = kwargs.pop("_maxlen", None)
        super().__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        if len(self.keys()) >= self._maxlen:
            self.popitem(last=False)
        super().__setitem__(key, value)

    @property
    def maxlen(self):
        return self._maxlen


@dataclass(frozen=True)
class ModelHandle:
    """Model handles for serving models."""

    spec: ModelSpec
    """Model specification."""
    handle: Union[ray.remote, ray.actor.ActorHandle]
    """Ray actor handle."""


class InferenceService(nos_service_pb2_grpc.InferenceServiceServicer):
    """
    Experimental gRPC-based inference service.

    This service is used to serve models over gRPC.

    Refer to the bring-your-own-schema section:
    https://docs.ray.io/en/master/serve/direct-ingress.html?highlight=grpc#bring-your-own-schema
    """

    def __init__(self):
        self.model_handle = FixedLengthFIFODict(_maxlen=4)
        self.executor = RayExecutor.get()
        try:
            self.executor.init()
        except Exception as e:
            logger.info(f"Failed to initialize executor: {e}")
            raise RuntimeError(f"Failed to initialize executor: {e}")

    def init_model(self, model_name: str):
        """Initialize the model."""
        assert model_name not in self.model_handle, f"Model already initialized: {model_name}"

        # Load the model spec
        try:
            spec = hub.load_spec(model_name)
            logger.debug(f"Loaded model spec: {spec}")
        except Exception as e:
            raise ModelNotFoundError(f"Failed to load model spec: {model_name}, {e}")

        # If the model handle is full, pop the oldest model
        if len(self.model_handle) >= self.model_handle.maxlen:
            spec = self.model_handle.popitem(last=False)
            self.delete_model(spec.model_name)
            logger.info(f"Deleting oldest model: {model_name}")
        logger.info(f"Initializing model: {model_name}")

        # Create the serve deployment from the model handle
        model_cls = spec.cls
        actor_options = {"num_gpus": 1 if torch.cuda.is_available() else 0}
        logger.debug(f"Creating actor: {actor_options}")
        actor_cls = ray.remote(**actor_options)(model_cls)
        # TOOD(spillai): Currently only one model per model-name is supported.
        self.model_handle[spec.name] = ModelHandle(spec, actor_cls.remote(*spec.args, **spec.kwargs))
        logger.info(f"Created actor: {self.model_handle[spec.name]}, type={type(self.model_handle[spec.name])}")
        logger.info(f"Models ({len(self.model_handle)}): {self.model_handle.keys()}")

    def delete_model(self, model_name: str):
        """Delete the model."""
        assert model_name in self.model_handle, f"Model not initialized: {model_name}"

        logger.info(f"Deleting model: {model_name}")
        model_handle = self.model_handle[model_name]
        ray.kill(model_handle.handle)
        self.model_handle.pop(model_name)
        logger.info(f"Deleted model: {model_name}")

    def ListModels(self, request: empty_pb2.Empty, context: grpc.ServicerContext) -> nos_service_pb2.ModelListResponse:
        """List all models."""
        return nos_service_pb2.ModelListResponse(models=hub.list())

    def InitModel(
        self, request: nos_service_pb2.InitModelRequest, context: grpc.ServicerContext
    ) -> nos_service_pb2.InitModelResponse:
        """Initialize the model."""
        if request.model_name in self.model_handle:
            return nos_service_pb2.InitModelResponse(result="ok")

        # Load the model spec
        try:
            self.init_model(request.model_name)
        except Exception as e:
            context.abort(context, grpc.StatusCode.NOT_FOUND, str(e))
        return nos_service_pb2.InitModelResponse(result="ok")

    def DeleteModel(
        self, request: nos_service_pb2.DeleteModelRequest, context: grpc.ServicerContext
    ) -> nos_service_pb2.DeleteModelResponse:
        """Delete the model."""
        self.delete_model(request.model_name)
        return nos_service_pb2.DeleteModelResponse(result="ok")

    def GetModelInfo(
        self, request: nos_service_pb2.ModelInfoRequest, context: grpc.ServicerContext
    ) -> nos_service_pb2.ModelInfoResponse:
        """Get model information."""
        raise NotImplementedError()

    def Predict(
        self, request: nos_service_pb2.InferenceRequest, context: grpc.ServicerContext
    ) -> nos_service_pb2.InferenceResponse:
        """Main model prediction interface."""
        logger.debug(f"Received request: {request.method}, {request.model_name}")
        if request.model_name not in self.model_handle:
            self.init_model(request.model_name)

        # TODO (spillai): This is inconsistent for CLIP which supports both (txt2vec, img2vec)
        # assert self.model_spec.method.value == request.method
        handle = self.model_handle.get(request.model_name).handle

        if request.method == MethodType.TXT2IMG.value:
            prompt = request.text_request.text
            logger.debug(f"Generating image with prompt: {prompt}")
            response_ref = handle.__call__.remote(prompt, height=512, width=512)
            (img,) = ray.get(response_ref)
            ref_bytes = cloudpickle.dumps({"image": img}, protocol=4)
            return nos_service_pb2.InferenceResponse(result=ref_bytes)

        elif request.method == MethodType.TXT2VEC.value:
            prompt = request.text_request.text
            logger.debug(f"Encoding text: {prompt}")
            response_ref = handle.encode_text.remote(prompt)
            embedding = ray.get(response_ref)
            ref_bytes = cloudpickle.dumps({"embedding": embedding}, protocol=4)
            return nos_service_pb2.InferenceResponse(result=ref_bytes)

        elif request.method == MethodType.IMG2VEC.value:
            img: Union[np.ndarray, Image.Image] = cloudpickle.loads(request.image_request.image_bytes)
            logger.debug(f"Encoding img (type={type(img)})")

            response_ref = handle.encode_image.remote(img)
            embedding = ray.get(response_ref)
            ref_bytes = cloudpickle.dumps({"embedding": embedding}, protocol=4)
            return nos_service_pb2.InferenceResponse(result=ref_bytes)

        elif request.method == MethodType.IMG2BBOX.value:
            img: Union[np.ndarray, Image.Image] = cloudpickle.loads(request.image_request.image_bytes)
            logger.debug(f"Encoding img (type={type(img)})")

            response_ref = handle.predict.remote(img)
            prediction = ray.get(response_ref)
            # prediction: {'scores': np.ndarray, 'labels': np.ndarray, 'bboxes': np.ndarray}
            ref_bytes = cloudpickle.dumps(prediction, protocol=4)
            return nos_service_pb2.InferenceResponse(result=ref_bytes)
        else:
            context.abort(context, grpc.StatusCode.INVALID_ARGUMENT, f"Invalid method {request.method}")


def serve(address: str = f"[::]:{DEFAULT_GRPC_PORT}", max_workers: int = 1) -> None:
    """Start the gRPC server."""
    from concurrent import futures

    options = [
        ("grpc.max_message_length", 512 * 1024 * 1024),
        ("grpc.max_send_message_length", 512 * 1024 * 1024),
        ("grpc.max_receive_message_length", 512 * 1024 * 1024),
    ]
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers), options=options)
    nos_service_pb2_grpc.add_InferenceServiceServicer_to_server(InferenceService(), server)
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
