import numpy as np
import tensorrt as trt
import torch
from torch_tensorrt.fx.converter_registry import tensorrt_converter
from torch_tensorrt.fx.converters.acc_ops_converters import _LOGGER, get_trt_plugin
from torch_tensorrt.fx.converters.converter_utils import get_trt_tensor
from torch_tensorrt.fx.tracer.acc_tracer.acc_normalizer import (
    register_acc_op,
    register_acc_op_mapping,
)
from torch_tensorrt.fx.utils import torch_dtype_from_trt


TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
trt.init_libnvinfer_plugins(TRT_LOGGER, "")
print("Register libnvinfer plugins")
plugin_registry = trt.get_plugin_registry()
print(f"Registry: {plugin_registry}")
for plugin in plugin_registry.plugin_creator_list:
    print(plugin.name)


@register_acc_op_mapping(
    op_and_target=("call_function", torch.nn.functional.group_norm),
    arg_replacement_tuples=[
        ("input", "input"),
        ("num_groups", "num_groups"),
        ("weight", "weight"),
        ("bias", "bias"),
        ("eps", "eps"),
    ],
)
@register_acc_op
def group_norm(*, input, num_groups, weight=None, bias=None, eps=1e-05):
    return torch.nn.functional.group_norm(input, num_groups, weight=weight, bias=bias, eps=eps)


@tensorrt_converter(group_norm)
def acc_ops_group_norm(network, target, args, kwargs, name):
    input_val = kwargs["input"]
    weight = kwargs["weight"]
    bias = kwargs["bias"]
    if weight is None:
        weight = torch.ones((*input_val.shape,)).to(torch_dtype_from_trt(input_val.dtype))
    weight = get_trt_tensor(network, weight, f"{name}_weight")

    if bias is None:
        bias = torch.zeros((*input_val.shape,)).to(torch_dtype_from_trt(input_val.dtype))
    bias = get_trt_tensor(network, bias, f"{name}_bias")

    if not isinstance(input_val, trt.tensorrt.ITensor):
        raise RuntimeError(f"GroupNorm received input {input_val} that is not part " "of the TensorRT region!")

    num_groups_field = trt.PluginField(
        "num_groups", np.array([kwargs["num_groups"]], dtype=np.int32), trt.PluginFieldType.INT32
    )
    eps_field = trt.PluginField("eps", np.array([kwargs["eps"]], dtype=np.float32), trt.PluginFieldType.FLOAT32)

    field_collection = trt.PluginFieldCollection([eps_field, num_groups_field])

    try:
        plugin = get_trt_plugin("GroupNormalizationPlugin", field_collection, "1", "")
        if plugin is None:
            raise Exception("Failed to build group norm plugin.")
    except AssertionError:
        _LOGGER.error("Unable to find group norm plugin, fall back to TensorRT implementation.")
        raise RuntimeError("Failed to build group norm plugin.")
    layer = network.add_plugin_v2([input_val, weight, bias], plugin)
    layer.name = name
    return layer.get_output(0)
