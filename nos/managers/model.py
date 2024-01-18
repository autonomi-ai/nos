"""Model manager for serving models with Ray actors."""
import gc
import os
import re
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Union

import humanize
import ray
import torch
from ray.runtime_env import RuntimeEnv
from ray.util.queue import Queue

from nos.common import ModelDeploymentSpec, ModelResources, ModelSpec, ModelSpecMetadataCatalog
from nos.logging import logger
from nos.managers.pool import ActorPool


NOS_MAX_CONCURRENT_MODELS = int(os.getenv("NOS_MAX_CONCURRENT_MODELS", 1))
NOS_MEMRAY_ENABLED = bool(int(os.getenv("NOS_MEMRAY_ENABLED", 0)))
NOS_RAY_LOGS_DIR = os.getenv("NOS_RAY_LOGS_DIR", "/tmp/ray/session_latest/logs")

if NOS_MEMRAY_ENABLED:
    try:
        import memray
    except ImportError:
        msg = "Failed to import memray, `pip install memray` before continuing."
        logger.error(msg)
        raise ImportError(msg)

    Path(NOS_RAY_LOGS_DIR).mkdir(parents=True, exist_ok=True)


class ModelResultQueue(Queue):
    """Ray-actor based queue for thread-safe put/get of model results."""

    def __init__(self, *args, **kwargs):
        """Initialize the results queue."""
        self._size = kwargs.pop("_maxsize", 1)
        super().__init__(*args, **kwargs)

    def ready(self) -> bool:
        """Check if the results queue is ready (i.e. full)."""
        return len(self) >= self._size

    def resize(self, size: int) -> None:
        """Resize the results queue."""
        assert not len(self), "Cannot resize queue when there are pending results."
        self._size = size


@dataclass
class ModelHandlePartial:
    """
    ModelHandle partial object with methods patched from model spec signature.

    Each method will have two variants:
        1. A callable function that can be used to call the method
            directly (e.g. `handle.process_images(images=images)`).
        2. A submit function that can be used to submit the method
            to the actor pool (e.g. `handle.submit_process_images(images=images)`).
    """

    handle: "ModelHandle"
    """Original model handle."""
    method: str
    """Method name."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.handle.__call__(*args, **kwargs, _method=self.method)

    def submit(self, *args: Any, **kwargs: Any) -> str:
        return self.handle.submit(*args, **kwargs, _method=self.method)


@dataclass
class _StreamingModelHandleResponse:
    iterable: Iterable[ray.ObjectRef]

    def __iter__(self):
        return self

    def __next__(self):
        return ray.get(next(self.iterable))


@dataclass
class ModelHandle:
    """Model handles for distributed model execution.

    Usage:
        ```python
        # Initialize a model handle
        >> model = ModelHandle(spec, num_replicas=1)

        # Call the task immediately
        >> response = model(**model_inputs)

        # Call a method on the model handle
        >> response = model.process_images(**model_inputs)

        # Submit a task to the model handle,
        # this will add results to the queue
        >> model.submit(**model_inputs)
        # Fetch the next result from the queue
        >> response = model.get()

        # Submit a task to a specific model handle method
        >> model.submit(**model_inputs, _method="process_images")

        # Submit a task to the model handle,
        # this will add results to the queue
        >> model_handle.submit(**model_inputs)
        # Fetch the next result from the queue
        >> response = model_handle.get()

        # Cleanup model resources
        >> model_handle.cleanup()
        ```
    """

    spec: ModelSpec
    """Model specification."""
    deployment: ModelDeploymentSpec = field(default_factory=ModelDeploymentSpec)
    """Number of replicas."""
    _actors: List[Union[ray.remote, ray.actor.ActorHandle]] = field(init=False, default=None)
    """Ray actor handle."""
    _actor_pool: ActorPool = field(init=False, default=None)
    """Ray actor pool."""
    _actor_options: Dict[str, Any] = field(init=False, default=None)
    """Ray actor options."""

    def __post_init__(self):
        """Initialize the actor handles."""
        self._actor_options = self._get_actor_options(self.spec, self.deployment)
        self._actors = [self._get_actor() for _ in range(self.deployment.num_replicas)]
        self._actor_pool = ActorPool(self._actors)

        # Patch the model handle with methods from the model spec signature
        for method in self.spec.signature:
            # Note (spillai): We do not need to patch the __call__ method
            # since it is already re-directed in the model handle.
            if hasattr(self, method):
                logger.debug(f"Model handle ({self}) already has method ({method}), skipping ....")
                continue

            # Methods:
            #   >> handle.process_images: ModelHandlePartial
            #   >> handle.process_images(images=...) => handle.__call__(images=..., _method="process_images")
            #   >> handle.process_images.submit(images=...) => handle.submit(images=..., _method="process_images")
            setattr(self, method, ModelHandlePartial(self, method))

    def __repr__(self) -> str:
        assert len(self._actors) == self.num_replicas
        opts_str = ", ".join([f"{k}={v}" for k, v in self._actor_options.items()])
        return f"ModelHandle(name={self.spec.name}, replicas={len(self._actors)}, opts=({opts_str}))"

    @property
    def num_replicas(self) -> int:
        """Get the number of replicas."""
        return self.deployment.num_replicas

    @classmethod
    def _get_actor_options(cls, spec: ModelSpec, deployment: ModelDeploymentSpec) -> Dict[str, Any]:
        """Get actor options from model specification."""
        # TOFIX (spillai): When considering CPU-only models with num_cpus specified,
        # OMP_NUM_THREADS will be set to the number of CPUs requested. Otherwise,
        # if num_cpus is not specified, OMP_NUM_THREADS will default to 1.
        # Instead, for now, we manually set the environment variable in `InferenceServiceRuntime`
        # to the number of CPUs threads available.

        # If deployment resources are not specified, get the model resources from the catalog
        if deployment.resources is None:
            try:
                catalog = ModelSpecMetadataCatalog.get()
                resources: ModelResources = catalog._resources_catalog[f"{spec.id}/{spec.default_method}"]
            except Exception:
                resources = ModelResources()
                logger.debug(f"Failed to get model resources [model={spec.id}, method={spec.default_method}]")

        # Otherwise, use the deployment resources provided
        else:
            resources = deployment.resources

        # For GPU models, we need to set the number of fractional GPUs to use
        if (resources.device == "auto" or resources.device == "gpu") and torch.cuda.is_available():
            try:
                # TODO (spillai): This needs to be resolved differently for
                # multi-node clusters.
                # Determine the current device id by checking the number of GPUs used.
                total, available = ray.cluster_resources(), ray.available_resources()
                gpus_used = total["GPU"] - available["GPU"]
                device_id = int(gpus_used)

                if isinstance(resources.device_memory, str) and resources.device_memory == "auto":
                    gpu_frac = 1.0 / NOS_MAX_CONCURRENT_MODELS
                    actor_opts = {"num_gpus": gpu_frac}
                elif isinstance(resources.device_memory, int):
                    # Fractional GPU memory needed within the current device
                    device_memory = torch.cuda.get_device_properties(device_id).total_memory
                    gpu_frac = float(resources.device_memory) / device_memory
                    gpu_frac = round(gpu_frac * 10) / 10.0

                    # Fractional GPU used for the current device
                    gpu_frac_used = gpus_used - int(gpus_used)
                    gpu_frac_avail = (1 - gpu_frac_used) * device_memory
                    logger.debug(
                        f"""actor_opts [model={spec.id}, """
                        f"""mem={humanize.naturalsize(resources.device_memory, binary=True)}, device={device_id}, device_mem={humanize.naturalsize(device_memory, binary=True)}, """
                        f"""gpu_frac={gpu_frac}, gpu_frac_avail={gpu_frac_avail}, """
                        f"""gpu_frac_used={gpu_frac_used}]"""
                    )
                    if gpu_frac > gpu_frac_avail:
                        logger.debug(
                            f"Insufficient GPU memory for model [model={spec.id}, "
                            f"method={spec.default_method}, gpu_frac={gpu_frac}, "
                            f"gpu_frac_avail={gpu_frac_avail}, gpu_frac_used={gpu_frac_used}]"
                        )
                        if device_id == torch.cuda.device_count() - 1:
                            # TOFIX (spillai): evict models to make space for the current model
                            logger.debug("All GPUs are fully utilized, this may result in undesirable behavior.")
                    actor_opts = {"num_gpus": gpu_frac}
                else:
                    raise ValueError(f"Invalid device memory: {resources.device_memory}")
            except Exception as exc:
                logger.debug(f"Failed to get GPU memory [e={exc}].")
                actor_opts = {"num_gpus": 1.0 / NOS_MAX_CONCURRENT_MODELS}

        elif resources.device == "cpu":
            actor_opts = {"num_cpus": resources.cpus, "memory": resources.memory}

        else:
            actor_opts = {"num_cpus": resources.cpus, "memory": resources.memory}

        if spec.runtime_env is not None:
            logger.debug("Using custom runtime environment, this may take a while to build.")
            actor_opts["runtime_env"] = RuntimeEnv(**asdict(spec.runtime_env))
        logger.debug(f"Actor options [id={spec.id}, opts={actor_opts}]")

        return actor_opts

    def _get_actor(self) -> Union[ray.remote, ray.actor.ActorHandle]:
        """Get an actor handle from model specification.

        Returns:
            Union[ray.remote, ray.actor.ActorHandle]: Ray actor handle.
        """
        # TODO (spillai): Use the auto-tuned model spec to instantiate an
        # actor the desired memory requirements. Fractional GPU amounts
        # will need to be calculated from the target HW and model spec
        # (i.e. 0.5 on A100 vs. T4 are different).
        # NOTE (spillai): Using default signature here is OK, since
        # all the signatures for a model spec have the same `func_or_cls`.
        model_cls = self.spec.default_signature.func_or_cls

        # Get the actor options from the model spec
        actor_options = self._actor_options
        actor_cls = ray.remote(**actor_options)(model_cls)

        # Check if the model class has the required method
        logger.debug(
            f"Creating actor [actor={actor_cls}, opts={actor_options}, cls={model_cls}, init_args={self.spec.default_signature.init_args}, init_kwargs={self.spec.default_signature.init_kwargs}]"
        )
        actor = actor_cls.remote(*self.spec.default_signature.init_args, **self.spec.default_signature.init_kwargs)

        # Note: Only check if default signature method is implemented
        # even though other methods may be implemented and used.
        if not hasattr(actor, self.spec.default_method):
            raise NotImplementedError(f"Model class {model_cls} does not have {self.spec.default_method} implemented.")
        logger.debug(f"Creating actor [actor={actor}, opts={actor_options}, cls={model_cls}]")

        # Add some memory logs to this actor
        if NOS_MEMRAY_ENABLED:
            # Replace all non-alphanumeric characters with underscores
            actor_name = re.sub(r"\W+", "_", str(actor))
            log_name = Path(NOS_RAY_LOGS_DIR) / f"{actor_name}_mem_profile.bin"
            if log_name.exists():
                log_name.unlink()
            try:
                memray.Tracker(log_name).__enter__()
            except Exception:
                logger.error("Failed to iniitialize memray tracker.")
        return actor

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call the task immediately.

        Args:
            *args: Model arguments.
            **kwargs: Model keyword arguments
                (except for special `_method` keyword that is
                used to call different class methods).
        Returns:
            Model response.
        """
        assert len(self._actors) >= 1, "Model should have atleast one replica."
        if self.num_replicas > 1:
            logger.warning("Model has >1 replicas, use `.submit()` instead to fully utilize them.")

        method: str = kwargs.pop("_method", self.spec.default_method)
        # TODO (spillai): We should be able to determine if the output
        # is an iterable or not from the signature, and set the default
        stream: bool = kwargs.pop("_stream", False)
        actor_method_func = getattr(self._actors[0], method)
        if not stream:
            response_ref: ray.ObjectRef = actor_method_func.remote(**kwargs)
            return ray.get(response_ref)
        else:
            response_refs: Iterable[ray.ObjectRef] = actor_method_func.options(num_returns="streaming").remote(
                **kwargs
            )
            return _StreamingModelHandleResponse(response_refs)

    def scale(self, num_replicas: Union[int, str] = 1) -> "ModelHandle":
        """Scale the model handle to a new number of replicas.

        Args:
            num_replicas (int or str): Number of replicas, or set to "auto" to
                automatically scale the model to the number of GPUs available.
        """
        if isinstance(num_replicas, str) and num_replicas == "auto":
            raise NotImplementedError("Automatic scaling not implemented.")
        if not isinstance(num_replicas, int):
            raise ValueError(f"Invalid replicas: {num_replicas}")

        # Check if there are any pending futures
        if self._actor_pool.has_next():
            logger.warning(f"Pending futures detected, this may result in dropped queue items [name={self.spec.name}]")
        logger.debug(f"Waiting for pending futures to complete before scaling [name={self.spec.name}].")
        logger.debug(f"Scaling model [name={self.spec.name}].")

        if num_replicas == len(self._actors):
            logger.debug(f"Model already scaled appropriately [name={self.spec.name}, replicas={num_replicas}].")
            return self
        elif num_replicas > len(self._actors):
            self._actors += [self._get_actor() for _ in range(num_replicas - len(self._actors))]
            logger.debug(f"Scaling up model [name={self.spec.name}, replicas={num_replicas}].")
        else:
            actors_to_remove = self._actors[num_replicas:]
            for actor in actors_to_remove:
                ray.kill(actor)
            self._actors = self._actors[:num_replicas]

            logger.debug(f"Scaling down model [name={self.spec.name}, replicas={num_replicas}].")

        # Update repicas and queue size
        self.deployment.num_replicas = num_replicas

        # Re-create the actor pool
        logger.debug(f"Removing actor pool [replicas={len(self._actors)}].")
        del self._actor_pool
        self._actor_pool = None

        # Re-create the actor pool
        logger.debug(f"Re-creating actor pool [name={self.spec.name}, replicas={num_replicas}].")
        self._actor_pool = ActorPool(self._actors)
        assert len(self._actors) == num_replicas, "Model scaling failed."
        gc.collect()
        return self

    def submit(self, *args: Any, **kwargs: Any) -> ray.ObjectRef:
        """Submit a task to the actor pool.

        Note (spillai): Caveats for `.submit()` with custom methods:
            ModelHandles have a single result queue that add
            results asynchronously on task completion. Calling `submit()`
            with different methods interchangably will result in
            the results queue being populated with results from
            different methods. In other words, it is advised to
            use `submit()` with the same method for a given model
            and then use `get()` to fetch all the results, before
            calling `submit()` with a different method.

        Args:
            *args: Model arguments.
            **kwargs: Model keyword arguments
                (except for special `_method` keyword that is
                used to call different class methods).

        Returns:
            ray.ObjectRef: Ray object reference as a string.
        """
        # Submit the task to the actor pool, leveraging all replicas
        method: str = kwargs.pop("_method", self.spec.default_method)
        # TODO (spillai): We should be able to determine if the output
        # is an iterable or not from the signature, and set the default
        stream: bool = kwargs.pop("_stream", False)
        remote_opts = {"num_returns": "streaming"} if stream else {}
        if not self._actor_pool._idle_actors:
            logger.warning(f"Actor pool is full, this may result in dropped queue items [name={self.spec.name}]")
        future_ref = self._actor_pool.submit(
            lambda a, v: getattr(a, method).options(**remote_opts).remote(**v), kwargs
        )
        logger.info(f"Submitted task [name={self.spec.name}, method={method}, kwargs={kwargs}]")
        return future_ref

    def cleanup(self) -> None:
        """Kill all the actor handles and garbage collect."""
        for actor_handle in self._actors:
            ray.kill(actor_handle)
        self._actors = []
        gc.collect()

    def get(self, future_ref: ray.ObjectRef = None, timeout: int = None) -> Any:
        """Get the result future."""
        return self._actor_pool.get(future_ref)

    async def async_get(self, future_ref: ray.ObjectRef = None, timeout: int = None) -> Any:
        """Get the result future asynchronously."""
        return await self._actor_pool.async_get(future_ref)


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

    max_concurrent_models: int = NOS_MAX_CONCURRENT_MODELS
    """Maximum number of concurrent models."""

    handlers: Dict[str, ModelHandle] = field(default_factory=OrderedDict)
    """Model handles mapped (key=model-identifier, value=ModelHandle)."""

    def __post_init__(self):
        """Initialize the model manager."""
        if self.policy not in (self.EvictionPolicy.FIFO,):
            raise NotImplementedError(f"Eviction policy not implemented: {self.policy}")
        if self.max_concurrent_models > 8:
            raise Exception(
                f"Large number of concurrent models requested, keep it <= 8 [concurrency={self.max_concurrent_models}]"
            )

    def __repr__(self) -> str:
        """String representation of the model manager (memory consumption, models, in tabular format etc)."""
        repr_str = f"\nModelManager(policy={self.policy}, models={len(self.handlers)})"
        for idx, (model_id, model_handle) in enumerate(self.handlers.items()):
            repr_str += f"\n  {idx}: [id={model_id}, model={model_handle}]"
        return repr_str

    def __len__(self) -> int:
        """Get the number of models in the manager."""
        return len(self.handlers)

    def __contains__(self, spec: ModelSpec) -> bool:
        """Check if a model exists in the manager.

        Args:
            spec (ModelSpec): Model specification.
        Returns:
            bool: True if the model exists, else False.
        """
        return spec.id in self.handlers

    def load(self, spec: ModelSpec, deployment: ModelDeploymentSpec = ModelDeploymentSpec()) -> ModelHandle:
        """Load a model handle from the manager using the model specification.

        Create a new model handle if it does not exist,
        else return an existing handle.

        Args:
            spec (ModelSpec): Model specification.
            deployment (ModelDeploymentSpec): Model deployment specification.
        Returns:
            ModelHandle: Model handle.
        """
        model_id: str = spec.id
        if model_id not in self.handlers:
            return self.add(spec, deployment)
        else:
            # Only scale the model if the number of replicas is specified,
            # otherwise treat it as a get without modifying the number of replicas.
            if deployment.num_replicas is not None and deployment.num_replicas != self.handlers[model_id].num_replicas:
                self.handlers[model_id].scale(deployment.num_replicas)
            return self.handlers[model_id]

    def get(self, spec: ModelSpec) -> ModelHandle:
        """Get a model handle from the manager using the model identifier.

        Args:
            spec (ModelSpec): Model specification.
        Returns:
            ModelHandle: Model handle.
        """
        model_id: str = spec.id
        if model_id not in self.handlers:
            return self.add(spec, ModelDeploymentSpec(num_replicas=1))
        else:
            return self.handlers[model_id]

    def add(self, spec: ModelSpec, deployment: ModelDeploymentSpec = ModelDeploymentSpec()) -> ModelHandle:
        """Add a model to the manager.

        Args:
            spec (ModelSpec): Model specification.
            deployment (ModelDeploymentSpec): Model deployment specification.
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

        # Create the serve deployment from the model handle
        # Note: Currently one model per (model-name, task) is supported.
        self.handlers[model_id] = ModelHandle(spec, deployment)
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
        logger.debug(self)
        assert model_id not in self.handlers, f"Model should have been evicted [model_id={model_id}]"
        return handle
