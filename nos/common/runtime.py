import importlib
from typing import Any, Dict, List

from pydantic.dataclasses import dataclass


def is_package_available(name: str) -> bool:
    """Check if a package is available."""
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        return False


def is_torch_tensorrt_available():
    return is_package_available("torch_tensorrt")


def is_torch_neuron_available():
    return is_package_available("torch_neuron")


def is_torch_neuronx_available():
    return is_package_available("torch_neuronx")


@dataclass
class RuntimeEnv:
    conda: Dict[str, Any]
    """Conda environment specification."""

    @classmethod
    def from_packages(cls, packages: List[str]) -> Dict[str, Any]:
        return cls(conda={"dependencies": ["pip", {"pip": packages}]})
