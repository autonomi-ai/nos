import os
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Union

import ray
import torch

from nos.common import ModelSpec
from nos.logging import logger


@dataclass
class ModelHandle:
    """Model handles for model execution.

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
    _actor_method_func: Union[ray.remote, ray.actor.ActorHandle] = field(init=False, default=None)
    """Ray actor method function."""

    def __post_init__(self):
        """Initialize the actor handles."""
        if self.num_replicas > 1:
            raise NotImplementedError("Multiple replicas not yet supported.")
        self._actors = [self.actor_from_spec(self.spec) for _ in range(self.num_replicas)]
        # Get the method function (i.e. `__call__`, or `predict`)
        try:
            self._actor_method_func = getattr(self.actor_handle, self.spec.signature.method_name)
        except AttributeError as exc:
            self._actor_method_func = None
            err = f"Failed to get method function: method={self.spec.signature.method_name}"
            logger.error(f"{err}, exc={exc}")
            raise Exception(err)

    @staticmethod
    def actor_from_spec(spec: ModelSpec) -> Union[ray.remote, ray.actor.ActorHandle]:
        """Get an actor handle from model specification.

        Args:
            spec (ModelSpec): Model specification.
        Returns:
            Union[ray.remote, ray.actor.ActorHandle]: Ray actor handle.
        """
        # TODO (spillai): Use the auto-tuned model spec to instantiate an
        # actor the desired memory requirements. Fractional GPU amounts
        # will need to be calculated from the target HW and model spec
        # (i.e. 0.5 on A100 vs. T4 are different).
        model_cls = spec.signature.func_or_cls
        actor_options = {"num_gpus": 0.1 if torch.cuda.is_available() else 0}
        logger.debug(f"Creating actor: {actor_options}, {model_cls}")
        actor_cls = ray.remote(**actor_options)(model_cls)
        return actor_cls.remote(*spec.signature.init_args, **spec.signature.init_kwargs)

    def kill(self) -> None:
        """Kill the actor handle."""
        for actor_handle in self._actors:
            ray.kill(actor_handle)

    def remote(self, *args, **kwargs) -> ray.ObjectRef:
        """Submit a task to the actor handle or pool.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        Returns:
            ray.ObjectRef: Ray object reference.
        """
        # Call the method function
        response_ref: ray.ObjectRef = self._actor_method_func.remote(**kwargs)
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

    max_concurrent_models: int = int(os.getenv("NOS_MAX_CONCURRENT_MODELS", 2))
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

    def __contains__(self, spec: ModelSpec) -> bool:
        """Check if a model exists in the manager.

        Args:
            spec (ModelSpec): Model specification.
        Returns:
            bool: True if the model exists, else False.
        """
        return spec.id in self.handlers

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

    def add(self, spec: ModelSpec) -> ModelHandle:
        """Add a model to the manager.

        Args:
            spec (ModelSpec): Model specification.
        Raises:
            ValueError: If the model already exists.
        Returns:
            ModelHandle: Model handle.
        """
        # If the model already exists, raise an error
        model_id = spec.id
        if model_id in self.handlers:
            raise ValueError(f"Model already exists [model_id={model_id}]")

        # If the model handle is full, pop the oldest model
        if len(self.handlers) >= self.max_concurrent_models:
            _handle: ModelHandle = self.evict()
            logger.debug(f"Deleting oldest model [model={_handle.spec.name}]")

        # Create the serve deployment from the model handle
        logger.debug(f"Initializing model with spec [model={spec.name}]")

        # Note: Currently one model per (model-name, task) is supported.
        self.handlers[model_id] = ModelHandle(spec)
        logger.debug(f"Created actor [handle={self.handlers[model_id]}, type={type(self.handlers[model_id])}]")
        logger.debug(f"Active models ({len(self.handlers)}): {list(self.handlers.keys())})")

        return self.handlers[model_id]

    def evict(self) -> ModelHandle:
        """Evict a model from the manager (FIFO, LRU etc).

        Returns:
            ModelHandle: Model handle.
        """
        # Pop the oldest model
        # TODO (spillai): Implement LRU policy
        assert len(self.handlers) > 0, "No models to evict."
        _, handle = self.handlers.popitem(last=False)
        model_id = handle.spec.id
        logger.debug(f"Deleting model [model_id={model_id}]")

        # Explicitly kill the model handle (including all actors)
        handle.kill()
        logger.debug(f"Deleted model [model_id={model_id}]")
        assert model_id not in self.handlers, f"Model should have been evicted [model_id={model_id}]"
        return handle
