from typing import Any, Callable, Dict, List, Optional, Type

from nos.common import FunctionSignature, ModelSpec, TaskType  # noqa: F401
from nos.hub.config import HuggingFaceHubConfig, MMLabConfig, NosHubConfig, TorchHubConfig  # noqa: F401
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
    def list(cls, private: bool = False) -> List[ModelSpec]:
        """List models in the registry.

        Args:
            private (bool): Whether to include private models.

        Returns:
            List[Dict[str, Any]]: List of model specifications.
        """
        if private:
            raise NotImplementedError("Private models not supported.")
        return [v for v in cls.get()._registry.values()]

    @classmethod
    def load_spec(cls, model_name: str, task: TaskType = None) -> ModelSpec:
        """Load model spec from the registry.

        Args:
            model_name (str): Model identifier (e.g. `openai/clip-vit-base-patch32`).
            task (TaskType): Task type (e.g. `TaskType.OBJECT_DETECTION_2D`).

        Returns:
            ModelSpec: Model specification.
        """
        try:
            return cls.get()._registry[ModelSpec.get_id(model_name, task=task)]
        except KeyError:
            raise KeyError(f"Unavailable model (name={model_name}).")

    @classmethod
    def load(cls, model_name: str, task: TaskType = None) -> Any:
        """Instantiate model from the registry.

        Args:
            model_name (str): Model identifier (e.g. `openai/clip-vit-base-patch32`).
            task (TaskType): Task type (e.g. `TaskType.OBJECT_DETECTION_2D`).

        Returns:
            Any: Instantiated model.
        """
        spec: ModelSpec = cls.load_spec(model_name, task=task)
        sig: FunctionSignature = spec.signature
        return sig.func_or_cls(*sig.init_args, **sig.init_kwargs)

    @classmethod
    def register(cls, model_name: str, task: TaskType, func_or_cls: Callable, **kwargs) -> ModelSpec:
        """Model registry decorator.

        Args:
            model_name (str): Model identifier (e.g. `openai/clip-vit-base-patch32`).
            task (TaskType): Task type (e.g. `TaskType.OBJECT_DETECTION_2D`).
            func_or_cls (Type[Any]): Model function or class.
            **kwargs: Additional keyword arguments.
        Returns:
            ModelSpec: Model specification.
        """
        spec = ModelSpec(
            name=model_name,
            task=task,
            signature=FunctionSignature(
                inputs=kwargs.pop("inputs", {}),
                outputs=kwargs.pop("outputs", {}),
                func_or_cls=func_or_cls,
                init_args=kwargs.pop("init_args", ()),
                init_kwargs=kwargs.pop("init_kwargs", {}),
                method_name=kwargs.pop("method_name", None),
            ),
        )
        model_id = ModelSpec.get_id(spec.name, spec.task)
        if model_id not in cls.get()._registry:
            cls.get()._registry[model_id] = spec
        return spec


# Alias methods
list = Hub.list
load = Hub.load
register = Hub.register
load_spec = Hub.load_spec
