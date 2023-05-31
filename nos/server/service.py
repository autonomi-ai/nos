from collections import OrderedDict
from dataclasses import dataclass
from typing import Union

import grpc
import ray
import rich.console
import rich.status
import torch
from google.protobuf import empty_pb2

from nos import hub
from nos.common import ModelSpec, TaskType, dumps
from nos.constants import DEFAULT_GRPC_PORT  # noqa F401
from nos.exceptions import ModelNotFoundError
from nos.executors.ray import RayExecutor
from nos.logging import logger
from nos.protoc import import_module
from nos.version import __version__


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


class InferenceServiceImpl(nos_service_pb2_grpc.InferenceServiceServicer):
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

    @logger.catch
    def init_model(self, model_name: str, task: TaskType = None):
        """Initialize the model."""
        assert model_name not in self.model_handle, f"Model already initialized: {model_name}"
        # Load the model spec
        try:
            spec: ModelSpec = hub.load_spec(model_name, task=task)
            logger.debug(f"Loaded model spec: {spec}")
        except Exception as e:
            raise ModelNotFoundError(f"Failed to load model spec: {model_name}, {e}")

        # If the model handle is full, pop the oldest model
        if len(self.model_handle) >= self.model_handle.maxlen:
            spec: ModelSpec = self.model_handle.popitem(last=False)
            self.delete_model(spec.name, task=spec.task)
            logger.info(f"Deleting oldest model: {model_name}")
        logger.info(f"Initializing model with spec: {spec}")

        # Create the serve deployment from the model handle
        model_cls = spec.signature.func_or_cls
        actor_options = {"num_gpus": 0.5 if torch.cuda.is_available() else 0}
        logger.debug(f"Creating actor: {actor_options}, {model_cls}")
        actor_cls = ray.remote(**actor_options)(model_cls)
        # Note: Currently one model per (model-name, task) is supported.
        self.model_handle[spec.id] = ModelHandle(
            spec, actor_cls.remote(*spec.signature.init_args, **spec.signature.init_kwargs)
        )
        logger.info(f"Created actor: {self.model_handle[spec.id]}, type={type(self.model_handle[spec.id])}")
        logger.info(f"Models ({len(self.model_handle)}): {self.model_handle.keys()}")

    @logger.catch
    def delete_model(self, model_str: str, task: TaskType = None):
        """Delete the model."""
        model_id = ModelSpec.get_id(model_str, task=task)
        assert model_id in self.model_handle, f"Model not initialized: {model_id}"

        logger.info(f"Deleting model: {model_id}")
        model_handle = self.model_handle[model_id]
        ray.kill(model_handle.handle)
        self.model_handle.pop(model_id)
        logger.info(f"Deleted model: {model_id}")

    @logger.catch
    def Ping(self, request: empty_pb2.Empty, context: grpc.ServicerContext) -> nos_service_pb2.PingResponse:
        """Health check."""
        return nos_service_pb2.PingResponse(status="ok")

    @logger.catch
    def GetServiceInfo(
        self, request: empty_pb2.Empty, context: grpc.ServicerContext
    ) -> nos_service_pb2.ServiceInfoResponse:
        """Get information on the service."""
        return nos_service_pb2.ServiceInfoResponse(version=__version__)

    @logger.catch
    def ListModels(self, request: empty_pb2.Empty, context: grpc.ServicerContext) -> nos_service_pb2.ModelListResponse:
        """List all models."""
        response = nos_service_pb2.ModelListResponse()
        for spec in hub.list():
            response.models.append(nos_service_pb2.ModelInfo(name=spec.name, task=spec.task.value))
        return response

    @logger.catch
    def GetModelInfo(
        self, request: nos_service_pb2.ModelInfoRequest, context: grpc.ServicerContext
    ) -> nos_service_pb2.ModelInfoResponse:
        """Get model information."""
        try:
            model_info = request.request
            spec: ModelSpec = hub.load_spec(model_info.name, task=TaskType(model_info.task))
        except KeyError as e:
            context.abort(context, grpc.StatusCode.NOT_FOUND, str(e))
        return spec._to_proto(public=True)

    @logger.catch
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
        ):
            context.abort(context, grpc.StatusCode.INVALID_ARGUMENT, f"Invalid task {model_request.task}")

        model_id = ModelSpec.get_id(model_request.name, task=TaskType(model_request.task))
        if model_id not in self.model_handle:
            self.init_model(model_request.name, TaskType(model_request.task))

        # Get the model handle (model_spec, actor_handle)
        model_handle: ModelHandle = self.model_handle[model_id]
        model_spec: ModelSpec = model_handle.spec

        # TODO (spillai): Validate the inputs
        model_inputs = model_spec.signature._decode_inputs(request.inputs)
        actor_handle = model_handle.handle

        # Get the method function (i.e. `__call__`, or `predict`)
        actor_method_func = getattr(actor_handle, model_spec.signature.method_name)
        logger.debug(f"Actor method function: {actor_method_func}")
        logger.debug(f"Actor handle: {actor_handle}")
        logger.debug(f"Model spec: {model_spec}")

        # Call the method function
        response_ref = actor_method_func.remote(**model_inputs)
        response = ray.get(response_ref)

        # If the response is a single value, wrap it in a dict with the appropriate key
        if len(model_spec.signature.outputs) == 1:
            response = {k: response for k in model_spec.signature.outputs}
        return nos_service_pb2.InferenceResponse(response_bytes=dumps(response))


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
