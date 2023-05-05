from typing import Any, Dict, List, Optional, Type

from nos.hub.config import HuggingFaceHubConfig, NosHubConfig, TorchHubConfig, MMLabConfig  # noqa: F401
from nos.hub.spec import MethodType, ModelSpec  # noqa: F401
from nos.logging import logger


class Hub:
    """Registry for models."""

    _instance: Optional["Hub"] = None
    """Singleton instance."""
    _registry: Dict[str, ModelSpec] = {}
    """Model specifications lookup for all models registered."""

    @classmethod
    def get(cls: "Hub") -> "Hub":
        """Get the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def list(cls, private: bool = False) -> List[Dict[str, Any]]:
        """List models in the registry."""
        if private:
            raise NotImplementedError("Private models not supported.")
        return [v.name for v in cls.get()._registry.values()]

    @classmethod
    def load_spec(cls, model_name: str) -> ModelSpec:
        """Load model spec from the registry."""
        try:
            return cls.get()._registry[model_name]
        except KeyError:
            raise KeyError(f"Unavailable model (name={model_name}).")

    @classmethod
    def load(cls, model_name: str) -> Any:
        """Instantiate model from the registry."""
        spec: ModelSpec = cls.load_spec(model_name)
        return spec.cls(*spec.args, **spec.kwargs)

    @classmethod
    def register(cls, model_name: str, method: str, model_cls: Type[Any], *args, **kwargs) -> Any:
        """Model registry decorator."""
        cls.get()._registry[model_name] = ModelSpec(
            name=model_name,
            method=MethodType[method.upper()],
            cls=model_cls,
            args=kwargs.pop("args", ()),
            kwargs=kwargs.pop("kwargs", {}),
        )
        logger.debug(f"Registered model: [name={model_name}]")


# Alias methods
list = Hub.list
load = Hub.load
register = Hub.register
load_spec = Hub.load_spec

# Register models / Populate the registry
import nos.models  # noqa: F401, E402
