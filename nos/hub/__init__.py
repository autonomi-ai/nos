from typing import Any, Callable, Dict, List, Optional, Type

from nos.common.metaclass import SingletonMetaclass  # noqa: F401
from nos.common.spec import (  # noqa: F401
    FunctionSignature,
    ModelSpec,
    ModelSpecMetadata,
    ModelSpecMetadataRegistry,
    TaskType,
)
from nos.hub.config import HuggingFaceHubConfig, MMLabConfig, MMLabHub, NosHubConfig, TorchHubConfig  # noqa: F401
from nos.hub.hf import hf_login  # noqa: F401
from nos.logging import logger


class Hub:
    """Registry for models."""

    _instance: Optional["Hub"] = None
    """Singleton instance."""
    _registry: Dict[str, ModelSpec] = {}
    """Model specifications lookup for all models registered."""
    _metadata_registry: ModelSpecMetadataRegistry = ModelSpecMetadataRegistry.get()
    """Model specification metadata registry."""

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
    def register(cls, model_id: str, task: TaskType, func_or_cls: Callable, **kwargs) -> ModelSpec:
        """Model registry decorator.

        Args:
            model_id (str): Model identifier (e.g. `openai/clip-vit-base-patch32`).
            task (TaskType): Task type (e.g. `TaskType.OBJECT_DETECTION_2D`).
            func_or_cls (Type[Any]): Model function or class.
            **kwargs: Additional keyword arguments.
        Returns:
            ModelSpec: Model specification.
        """
        logger.debug(f"Registering model: {model_id}")
        spec = ModelSpec(
            model_id,
            signature=FunctionSignature(
                func_or_cls,
                inputs=kwargs.pop("inputs", {}),
                outputs=kwargs.pop("outputs", {}),
                init_args=kwargs.pop("init_args", ()),
                init_kwargs=kwargs.pop("init_kwargs", {}),
                method=kwargs.pop("method", "__call__"),
            ),
        )

        hub = cls.get()
        if model_id not in hub._metadata_registry:
            hub._metadata_registry[model_id] = ModelSpecMetadata(model_id, task)
        if model_id not in hub._registry:
            hub._registry[model_id] = spec
        logger.debug(f"Registered model [id={model_id}, spec={spec}]")
        return spec


# Alias methods
list = Hub.list
load = Hub.load
register = Hub.register
load_spec = Hub.load_spec
