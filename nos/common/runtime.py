def is_torch_tensorrt_available():
    try:
        import torch_tensorrt  # noqa: F401

        return True
    except ImportError:
        return False


def is_torch_neuron_available():
    try:
        import torch_neuron  # noqa: F401

        return True
    except ImportError:
        return False


def is_torch_neuronx_available():
    try:
        import torch_neuronx  # noqa: F401

        return True
    except ImportError:
        return False
