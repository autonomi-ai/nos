import importlib


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
