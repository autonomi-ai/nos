import torch


def get_submod_inputs(_mod, _submod, _inputs):
    """Get inputs of a model by registering a forward_pre_hook on the callable."""
    acc_inputs = None

    def get_input(self, __inputs):
        nonlocal acc_inputs
        acc_inputs = __inputs

    handle = _submod.register_forward_pre_hook(get_input)
    with torch.inference_mode():
        _mod(*_inputs)
    handle.remove()
    return acc_inputs
