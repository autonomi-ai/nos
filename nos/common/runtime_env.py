from dataclasses import dataclass, field
from typing import Any, Dict, List

from nos.common.metaclass import SingletonMetaclass
from nos.logging import logger


@dataclass
class RuntimeEnv:
    conda: Dict[str, Any]
    """Conda environment specification."""

    @classmethod
    def from_packages(cls, packages: List[str], **kwargs) -> Dict[str, Any]:
        return cls(conda={"dependencies": ["pip", {"pip": packages}]}, **kwargs)


@dataclass
class RuntimeEnvironmentsHub(metaclass=SingletonMetaclass):
    """Singleton class for managing conda runtime environments."""

    _registry: Dict[str, RuntimeEnv] = field(init=False, default_factory=dict)
    """Runtime environments registry."""

    @classmethod
    def register(cls, name: str, runtime_env: RuntimeEnv) -> None:
        """Register a runtime environment."""
        cls()._registry[name] = runtime_env
        logger.debug(f"Registered runtime environment (name={name}).")

    @classmethod
    def get(cls, name: str) -> RuntimeEnv:
        """Get a runtime environment."""
        return cls()._registry[name]

    @classmethod
    def list(cls) -> List[str]:
        """List all registered runtime environments."""
        return list(cls()._registry.keys())

    def __contains__(self, name: str) -> bool:
        """Check if a runtime environment is registered."""
        return name in self._registry

    def __getitem__(self, name: str) -> RuntimeEnv:
        """Get a runtime environment."""
        return self._registry[name]


def register(name: str, runtime_env: RuntimeEnv) -> None:
    """Register a runtime environment."""
    RuntimeEnvironmentsHub.register(name, runtime_env)
