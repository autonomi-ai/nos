import os
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Union

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


@dataclass
class ModelHandle:
    """Model handles for serving models.

    Usage:
        ```python
        # Initialize a model handle
        >> model_handle = ModelHandle(spec, num_replicas=1)

        # Submit a task to the model handle
        >> handler = model_handle.handle
        >> response_ref = handler.submit(**model_inputs)

        # Kill all actors
        >> model_handle.kill()
        ```
    """

    spec: ModelSpec
    """Model specification."""
    num_replicas: Union[int, str] = field(default=1)
    """Number of replicas."""

    _actors: List[Union[ray.remote, ray.actor.ActorHandle]] = field(init=False, default=None)
    """Ray actor handle."""

    def __post_init__(self):
        """Initialize the actor handles."""
        if self.num_replicas > 1:
            raise NotImplementedError("Multiple replicas not yet supported.")
        self._actors = [self.actor_from_spec(self.spec) for _ in range(self.num_replicas)]

    @staticmethod
    def actor_from_spec(spec: ModelSpec) -> Union[ray.remote, ray.actor.ActorHandle]:
        """Get an actor handle from model specification."""
        # TODO (spillai): Use the auto-tuned model spec to instantiate an
        # actor the desired memory requirements. Fractional GPU amounts
        # will need to be calculated from the target HW and model spec
        # (i.e. 0.5 on A100 vs. T4 are different).
        model_cls = spec.signature.func_or_cls
        actor_options = {"num_gpus": 0.5 if torch.cuda.is_available() else 0}
        logger.debug(f"Creating actor: {actor_options}, {model_cls}")
        actor_cls = ray.remote(**actor_options)(model_cls)
        return actor_cls.remote(*spec.signature.init_args, **spec.signature.init_kwargs)

    def kill(self) -> None:
        """Kill the actor handle."""
        for actor_handle in self._actors:
            ray.kill(actor_handle)

    def remote(self, *args, **kwargs) -> ray.ObjectRef:
        """Submit a task to the actor handle or pool."""
        # Get the method function (i.e. `__call__`, or `predict`)
        try:
            actor_method_func = getattr(self.actor_handle, self.spec.signature.method_name)
        except AttributeError as exc:
            err = f"Failed to get method function: method={self.spec.signature.method_name}"
            logger.error(f"{err}, exc={exc}")
            raise Exception(err)

        # Call the method function
        response_ref: ray.ObjectRef = actor_method_func.remote(**kwargs)
        return ray.get(response_ref)

    @property
    def actor_handle(self) -> Union[ray.remote, ray.actor.ActorHandle]:
        """Get the actor handle."""
        assert len(self._actors) == 1, "Only one actor handle is supported."
        return self._actors[0]


@dataclass(frozen=True)
class ModelManager:
    """Model manager for serving models with ray actors.

    Features:
      * Concurrency: Support fixed number of concurrent models,
        running simultaneously with FIFO model eviction policies.
      * Parallelism: Support multiple replicas of the same model.
      * Optimal memory management: Model memory consumption
        are automatically inferred from the model specification
        and used to optimally bin-pack models on the GPU.
      * Automatic garbage collection: Models are automatically
        garbage collected when they are evicted from the manager.
        Scaling models with the model manager should not result in OOM.

    """

    class EvictionPolicy(str, Enum):
        FIFO = "FIFO"
        LRU = "LRU"

    policy: EvictionPolicy = EvictionPolicy.FIFO
    """Eviction policy."""

    max_concurrent_models: int = os.getenv("NOS_MAX_CONCURRENT_MODELS", 4)
    """Maximum number of concurrent models."""

    handlers: Dict[str, ModelHandle] = field(default_factory=OrderedDict)
    """Model handles."""

    def __post_init__(self):
        if self.policy not in (self.EvictionPolicy.FIFO,):
            raise NotImplementedError(f"Eviction policy not implemented: {self.policy}")

    def __repr__(self) -> str:
        """String representation of the model manager (memory consumption, models, in tabular format etc)."""
        repr_str = f"ModelManager(policy={self.policy}, len(handlers)={len(self.handlers)})"
        for idx, (model_id, model_handle) in enumerate(self.handlers.items()):
            repr_str += f"\n\t{idx}: {model_id}, {model_handle}"
        return repr_str

    def get(self, model_spec: ModelSpec) -> ModelHandle:
        """Get a model handle from the manager.

        Create a new model handle if it does not exist,
        else return an existing handle.

        Args:
            model_spec (ModelSpec): Model specification.
        Returns:
            ModelHandle: Model handle.
        """
        model_id: str = model_spec.id
        if model_id not in self.handlers:
            return self.add(model_spec)
        else:
            return self.handlers[model_id]

    @logger.catch
    def add(self, spec: ModelSpec) -> ModelHandle:
        """Add a model to the manager.

        Args:
            spec (ModelSpec): Model specification.
        Returns:
            ModelHandle: Model handle.
        """
        # If the model already exists, raise an error
        model_id = spec.id
        if model_id in self.handlers:
            raise ValueError(f"Model already exists: {model_id}")

        # If the model handle is full, pop the oldest model
        if len(self.handlers) >= self.max_concurrent_models:
            _handle: ModelHandle = self.evict()
            logger.info(f"Deleting oldest model: {_handle.spec.name}")

        # Create the serve deployment from the model handle
        logger.info(f"Initializing model with spec: {spec.name}")

        # Note: Currently one model per (model-name, task) is supported.
        self.handlers[model_id] = ModelHandle(spec)
        logger.info(f"Created actor: {self.handlers[model_id]}, type={type(self.handlers[model_id])}")
        logger.info(f"Models ({len(self.handlers)}): {self.handlers.keys()}")

        return self.handlers[model_id]

    @logger.catch
    def evict(self) -> ModelHandle:
        """Evict a model from the manager (FIFO, LRU etc)."""
        # Pop the oldest model
        # TODO (spillai): Implement LRU policy
        assert len(self.handlers) > 0, "No models to evict."
        _, handle = self.handlers.popitem(last=False)
        model_id = handle.spec.id
        logger.info(f"Deleting model: {model_id}")

        # Explicitly kill the model handle (including all actors)
        handle.kill()
        logger.info(f"Deleted model: {model_id}")
        assert model_id not in self.handlers, f"Model should have been evicted: {model_id}"
        return handle


class InferenceService:
    """Ray-executor based inference service.

    Parameters:
        model_manager (ModelManager): Model manager.
        executor (RayExecutor): Ray executor.

    Note: To be used with the `InferenceServiceImpl` gRPC service.
    """

    def __init__(self):
        self.model_manager = ModelManager()
        self.executor = RayExecutor.get()
        try:
            self.executor.init()
        except Exception as e:
            logger.info(f"Failed to initialize executor: {e}")
            raise RuntimeError(f"Failed to initialize executor: {e}")

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
        model_inputs = model_spec.signature._decode_inputs(inputs)

        # Initialize the model (if not already initialized)
        # This call should also evict models and garbage collect if
        # too many models are loaded are loaded simultaneously.
        model_handle: ModelHandle = self.model_manager.get(model_spec)

        # Get the model handle and call it remotely (with model spec, actor handle)
        response = model_handle.remote(**model_inputs)

        # If the response is a single value, wrap it in a dict with the appropriate key
        if len(model_spec.signature.outputs) == 1:
            response = {k: response for k in model_spec.signature.outputs}

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
            TaskType.IMAGE_SEGMENTATION_2D.value,
        ):
            context.abort(context, grpc.StatusCode.NOT_FOUND, f"Invalid task {model_request.task}")

        try:
            logger.debug(f"Executing request: {model_request.task}, {model_request.name}")
            response = self.execute(model_request.name, task=TaskType(model_request.task), inputs=request.inputs)
            return nos_service_pb2.InferenceResponse(response_bytes=dumps(response))
        except Exception as exc:
            logger.error(f"Failed to execute request: {model_request.task}, {model_request.name}, {request.inputs}")
            context.abort(context, grpc.StatusCode.INTERNAL, str(exc))


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
