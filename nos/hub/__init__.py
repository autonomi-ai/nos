from dataclasses import dataclass
from functools import wraps
from typing import Any, Dict, List, Optional

from nos.logging import logger


@dataclass(frozen=True)
class TorchHubConfig:
    repo: str
    model_name: str
    checkpoint: str = None


class Hub:
    """Registry for models."""

    _instance: Optional["Hub"] = None
    _configs = {}

    @classmethod
    def get(cls) -> "Hub":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def list(cls, private: bool = False) -> List[Dict[str, Any]]:
        """List models in the registry."""
        if private:
            raise NotImplementedError("Private models not supported.")
        return [v["name"] for v in cls.get()._configs.values()]

    @classmethod
    def load(cls, model_name: str) -> Any:
        """Load a model from the registry."""
        try:
            config = cls.get()._configs.get(model_name)
            model_cls = config["model_cls"]
            return model_cls()
        except KeyError:
            raise KeyError(f"Unavailable model (name={model_name}).")

    @classmethod
    def register(cls, model_name: str, model_cls: Any) -> None:
        """Model registry decorator."""
        cls.get()._configs[model_name] = {
            "name": model_name,
            "model_cls": model_cls,
        }
        logger.debug(f"Registered model: [name={model_name}]")
        return model_cls


def list(private: bool = False) -> List[Dict[str, Any]]:
    """List models in the registry."""
    if private:
        raise NotImplementedError("Private models not supported.")
    return Hub.list()


def load(model_name: str) -> Any:
    """Load a model from the registry."""
    try:
        return Hub.load(model_name)
    except KeyError:
        raise KeyError(f"Unavailable model (name={model_name}).")


def register(model_name: str) -> Any:
    """Model registry decorator."""

    @wraps(register)
    def _register(model_cls: Any) -> Any:
        Hub.register(model_name, model_cls)
        return model_cls

    return _register


# Register models
import nos.models  # noqa: F401, E402
