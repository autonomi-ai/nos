import gc
import os
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Union

import ray
import torch
from ray.runtime_env import RuntimeEnv
from ray.util.queue import Queue

from nos.common import ModelSpec
from nos.logging import logger


NOS_MEMRAY_ENABLED = os.getenv("NOS_MEMRAY_ENABLED")


@dataclass
class ModelHandle:
    """Model handles for model execution.

    Usage:
        ```python
        # Initialize a model handle
        >> model_handle = ModelHandle(spec, num_replicas=1)

        # Call the task immediately
        >> response = model_handle(**model_inputs)

        # Submit a task to the model handle
        >> response_ref = model_handle.submit(**model_inputs)

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
    _actor_pool: ray.util.ActorPool = field(init=False, default=None)
    """Ray actor pool."""
    _results_queue: Queue = field(init=False, default_factory=Queue)
    """Queue to fetch results from the actor pool."""
    _results_queue_size: int = field(init=False, default=None)
    """Maximum results queue size."""

    def __post_init__(self):
        """Initialize the actor handles."""
        self._actors = [self.get_actor(self.spec) for _ in range(self.num_replicas)]
        self._actor_pool = ray.util.ActorPool(self._actors)
        self._results_queue_size = 2 * self.num_replicas

    def __repr__(self) -> str:
        assert len(self._actors) == self.num_replicas
        return f"ModelHandle(name={self.spec.name}, replicas={len(self._actors)}, qsize={self._results_queue_size})"

    def scale(self, replicas: Union[int, str] = 1) -> "ModelHandle":
        """Scale the model handle to a new number of replicas.

        Args:
            replicas (int or str): Number of replicas, or set to "auto" to
                automatically scale the model to the number of GPUs available.
        """
        if isinstance(replicas, str) and replicas == "auto":
            raise NotImplementedError("Automatic scaling not implemented.")
        if not isinstance(replicas, int):
            raise ValueError(f"Invalid replicas: {replicas}")

        if replicas == len(self._actors):
            logger.debug(f"Model already scaled appropriately [name={self.spec.name}, replicas={replicas}].")
            return self
        elif replicas > len(self._actors):
            self._actors += [self.get_actor(self.spec) for _ in range(replicas - len(self._actors))]
            logger.debug(f"Scaling up model [name={self.spec.name}, replicas={replicas}].")
        else:
            actors_to_remove = self._actors[replicas:]
            for actor in actors_to_remove:
                ray.kill(actor.actor)
            self._actors = self._actors[:replicas]

            logger.debug(f"Scaling down model [name={self.spec.name}, replicas={replicas}].")

        # Update repicas and queue size
        self.num_replicas = replicas
        self._results_queue_size = 2 * self.num_replicas

        # Re-create the actor pool
        del self._actor_pool
        self._actor_pool = ray.util.ActorPool(self._actors)
        assert len(self._actors) == replicas, "Model scaling failed."
        gc.collect()
        return self

    @classmethod
    def _actor_options(cls, spec: ModelSpec) -> Dict[str, Any]:
        """Get actor options from model specification."""
        return {"num_gpus": 0.1 if torch.cuda.is_available() else 0}

    @classmethod
    def get_actor(cls, spec: ModelSpec) -> Union[ray.remote, ray.actor.ActorHandle]:
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
        if spec.runtime_env is not None:
            logger.debug("Using custom runtime environment, this may take a while to build.")
            actor_options["runtime_env"] = RuntimeEnv(**asdict(spec.runtime_env))
        logger.debug(f"Creating actor: {actor_options}, {model_cls}")

        # Add some memory logs to this actor
        actor_options = cls._actor_options(spec)
        logger.debug(f"Creating model handle [name={model_cls.__name__}, opts={actor_options}]")
        actor_cls = ray.remote(**actor_options)(model_cls)
        if NOS_MEMRAY_ENABLED:
            import memray

            flattened_name = spec.name.replace("/", "_")
            log_name = "/tmp/ray/session_latest/logs/" f"{flattened_name}_mem_profile.bin"
            if os.path.exists(log_name):
                os.remove(log_name)
            try:
                memray.Tracker(log_name).__enter__()
            except Exception:
                print("Tracker may have already been initialized, skipping...")
        return actor_cls.remote(*spec.signature.init_args, **spec.signature.init_kwargs)

    def cleanup(self) -> None:
        """Kill all the actor handles and garbage collect."""
        for actor_handle in self._actors:
            ray.kill(actor_handle)
        self._actors = []
        gc.collect()

    def __call__(self, *args, **kwargs) -> Any:
        """Call the task immediately.

        Args:
            *args: Model arguments.
            **kwargs: Model keyword arguments.
        Returns:
            ray.ObjectRef: Ray object reference.
        """
        if self.num_replicas > 1:
            logger.warning("Model has >1 replicas, use `.submit()` instead to fully utilize them.")
        self.submit(*args, **kwargs)
        self._fetch_next()
        return self.get()

    def submit(self, *args, **kwargs) -> None:
        """Submit a task to the actor pool.

        Args:
            *args: Model arguments.
            **kwargs: Model keyword arguments.
        """
        assert not len(self._actor_pool._pending_submits), "Pending submits should be empty."

        # Submit the task to the actor pool, leveraging all replicas
        self._actor_pool.submit(lambda a, v: getattr(a, self.spec.signature.method_name).remote(**v), kwargs)

        # If there are pending submissions due to the actor pool being fully utilized,
        # fetch the next result from the actor pool and put it in the queue.
        if len(self._actor_pool._pending_submits):
            self._fetch_next()
        assert not len(self._actor_pool._pending_submits), "Pending submits should be empty."

    def has_next(self) -> bool:
        """Check if the handle has a result in the queue."""
        return self._actor_pool.has_next() or len(self._results_queue)

    def get(self) -> Any:
        """Get the next result from the actor pool queue or by the object reference."""
        if not len(self._results_queue):
            self._results_queue.put(self._actor_pool.get_next())
        return self._results_queue.get()

    def full(self) -> bool:
        """Check if the results queue is full."""
        return len(self._results_queue) >= self._results_queue_size

    def _fetch_next(self) -> None:
        """Fetch results from the actor pool."""
        self._results_queue.put(self._actor_pool.get_next())
        if len(self._results_queue) > self._results_queue_size:
            logger.warning("Results queue full, dropping results. Use `.get()` to get results.")
            self._results_queue.get()

    @property
    def pending(self) -> List[ray.ObjectRef]:
        """Get the pending submisions."""
        return self._actor_pool._pending_submits

    @property
    def results(self) -> Queue:
        """Get the results queue."""
        return self._results_queue


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

    max_concurrent_models: int = int(os.getenv("NOS_MAX_CONCURRENT_MODELS", 1))
    """Maximum number of concurrent models."""

    handlers: Dict[str, ModelHandle] = field(default_factory=OrderedDict)
    """Model handles."""

    def __post_init__(self):
        if self.policy not in (self.EvictionPolicy.FIFO,):
            raise NotImplementedError(f"Eviction policy not implemented: {self.policy}")

    def __repr__(self) -> str:
        """String representation of the model manager (memory consumption, models, in tabular format etc)."""
        repr_str = f"\nModelManager(policy={self.policy}, models={len(self.handlers)})"
        for idx, (model_id, model_handle) in enumerate(self.handlers.items()):
            repr_str += f"\n  {idx}: [id={model_id}, model={model_handle}]"
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
        # Note: Currently one model per (model-name, task) is supported.
        self.handlers[model_id] = ModelHandle(spec)
        logger.debug(f"Added model [{self.handlers[model_id]}]")
        logger.debug(self)

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

        # Explicitly cleanup the model handle (including all actors)
        handle.cleanup()
        logger.debug(f"Deleted model [model_id={model_id}]")
        assert model_id not in self.handlers, f"Model should have been evicted [model_id={model_id}]"
        return handle
