from typing import Any, Callable, Dict, List, Optional, Tuple, Type

from nos.common.metaclass import SingletonMetaclass  # noqa: F401
from nos.common.spec import (  # noqa: F401
    FunctionSignature,
    ModelSpec,
    ModelSpecMetadata,
    ModelSpecMetadataRegistry,
    TaskType,
)
from nos.hub.config import HuggingFaceHubConfig, NosHubConfig, TorchHubConfig  # noqa: F401
from nos.hub.hf import hf_login  # noqa: F401
from nos.logging import logger


class Hub:
    """Registry for models."""

    _instance: Optional["Hub"] = None
    """Singleton instance."""
    _registry: Dict[str, ModelSpec] = {}
    """Model specifications lookup for all models registered."""

    @classmethod
    def get(cls: "Hub") -> "Hub":
        """Get the singleton instance.

        Returns:
            Hub: Singleton instance.
        """
        if cls._instance is None:
            cls._instance = cls()
            # Register models / Populate the registry
            import nos.models  # noqa: F401, E402
        return cls._instance

    @classmethod
    def list(cls, private: bool = False) -> List[str]:
        """List models in the registry.

        Args:
            private (bool): Whether to include private models.

        Returns:
            List[str]: List of model names.
        """
        if private:
            raise NotImplementedError("Private models not supported.")
        return [k for k in cls.get()._registry.keys()]

    @classmethod
    def load_spec(cls, model_id: str) -> ModelSpec:
        """Load model spec from the registry.

        Args:
            model_id (str): Model identifier (e.g. `openai/clip-vit-base-patch32`).

        Returns:
            ModelSpec: Model specification.
        """
        try:
            return cls.get()._registry[model_id]
        except KeyError:
            raise KeyError(f"Unavailable model (name={model_id}).")

    @classmethod
    def load(cls, model_id: str) -> Any:
        """Instantiate model from the registry.

        Args:
            model_id (str): Model identifier (e.g. `openai/clip-vit-base-patch32`).

        Returns:
            Any: Instantiated model.
        """
        spec: ModelSpec = cls.load_spec(model_id)
        # Note (spillai): Loading the default signature here is OK
        # since all the signatures have the same `func_or_cls`.
        sig: FunctionSignature = spec.default_signature
        return sig.func_or_cls(*sig.init_args, **sig.init_kwargs)

    @classmethod
    def register(
        cls,
        model_id: str,
        task: TaskType,
        func_or_cls: Callable,
        method: str = "__call__",
        init_args: Tuple[Any] = (),
        init_kwargs: Dict[str, Any] = {},  # noqa: B006
        inputs: Dict[str, Any] = {},  # noqa: B006
        outputs: Dict[str, Any] = {},  # noqa: B006
        **kwargs,
    ) -> ModelSpec:
        """Model registry decorator.

        Args:
            model_id (str): Model identifier (e.g. `openai/clip-vit-base-patch32`).
            task (TaskType): Task type (e.g. `TaskType.OBJECT_DETECTION_2D`).
            func_or_cls (Type[Any]): Model function or class.
            **kwargs: Additional keyword arguments.
        Returns:
            ModelSpec: Model specification.
        """
        logger.debug(
            f"""Registering model [model={model_id}, task={task}, func_or_cls={func_or_cls}, """
            f"""inputs={inputs}, outputs={outputs}, """
            f"""init_args={init_args}, init_kwargs={init_kwargs}, method={method}]"""
        )

        # Create signature
        signature: Dict[str, FunctionSignature] = {
            method: FunctionSignature(
                func_or_cls,
                method=method,
                init_args=init_args,
                init_kwargs=init_kwargs,
                input_annotations=inputs,
                output_annotations=outputs,
            ),
        }
        # Add metadata for the model
        metadata: Dict[str, ModelSpecMetadata] = {
            method: ModelSpecMetadata(model_id, method, task),
        }
        spec = ModelSpec(model_id, signature=signature, _metadata=metadata)
        logger.debug(f"Created model spec [id={model_id}, spec={spec}, metadata={metadata}]")

        # Get hub instance
        hub = cls.get()

        # Register model id to model spec registry
        if model_id not in hub._registry:
            hub._registry[model_id] = spec
            logger.debug(f"Registered model to hub registry [id={model_id}, spec={spec}]")

        # Add another signature if the model is already registered
        else:
            _spec = hub._registry[model_id]
            if method not in _spec.signature:
                logger.debug(
                    f"Adding task signature [model={model_id}, task={task}, method={method}, sig={spec.signature}]"
                )
                _spec.signature[method] = spec.signature[method]
                _spec._metadata[method] = spec._metadata[method]
            else:
                logger.debug(
                    f"Task signature already registered [model={model_id}, task={task}, method={method}, sig={spec.signature}]"
                )

        logger.debug(f"Registered model [id={model_id}, spec={spec}]")
        return spec


# Alias methods
list = Hub.list
load = Hub.load
register = Hub.register
load_spec = Hub.load_spec
