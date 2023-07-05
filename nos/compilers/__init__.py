import contextlib
import io
import time
import traceback
from dataclasses import dataclass
from typing import Any, Dict

import torch
from torch.fx import Tracer
from transformers.utils.fx import HFTracer

from nos.logging import logger


NOSTracer = HFTracer


@dataclass(frozen=True)
class NOSCompilerOptions:
    """Options for the NOS compiler."""

    verbose: bool = False
    """Whether to print verbose logs."""
    tracer_cls: torch.fx.Tracer = NOSTracer
    """Tracer class to use for tracing."""
    return_intermediates: bool = False
    """Whether to return intermediate results
    such as traced model and compiled model."""


@dataclass(frozen=True)
class NOSCompilerOutput:
    args: Dict[str, Any]
    """Arguments used for tracing."""
    concrete_args: Dict[str, Any]
    """Concrete arguments used for tracing."""
    precision: torch.dtype
    """Precision of the traced model."""
    options: NOSCompilerOptions
    """Tracer class used for tracing."""
    traced_model: torch.nn.Module = None
    """Traced model."""
    compiled_model: torch.nn.Module = None
    """Compiled model."""


def compile(
    model: torch.nn.Module,
    cargs: Dict[str, Any],
    concrete_args: Dict[str, Any] = None,
    precision: torch.dtype = torch.float32,
    options: NOSCompilerOptions = NOSCompilerOptions(),
    slug: str = None,
) -> torch.nn.Module:
    """ "Compile a model with NOS.

    Args:
        model: Model to compile.
        args: Arguments to use for tracing.
        concrete_args: Concrete arguments to use for tracing.
        precision: Precision to use for tracing.
        options: Options for the NOS compiler.

    Returns:
        torch.nn.Module: Compiled model.
    """
    assert isinstance(model, torch.nn.Module)
    assert precision in (torch.float32, torch.float16), "Precision must be one of: float32, float16"
    assert options.tracer_cls in (
        NOSTracer,
        HFTracer,
        Tracer,
    ), "Tracer must be one of: NOSTracer, huggingface.fx.HFTracer, torch.fx.Tracer"
    assert torch.cuda.is_available(), "Cannot compile model without CUDA support"

    import torch_tensorrt
    from torch_tensorrt.fx.utils import LowerPrecision

    def log_mem_usage():
        logger.debug(
            f"Memory usage (device | free | total): {device_id} | {free / 1024 ** 3:.1f} | {total / 1024 ** 3:.1f} GiB"
        )

    st = time.time()
    if slug is None:
        model_cls = model.__class__
        slug = f"{model_cls.__module__}.{model_cls.__name__}"

    logger.debug("Validating inputs")
    assert isinstance(cargs, dict), "Arguments must be a dictionary"
    assert len(cargs) > 0, "Arguments must not be empty"
    # TODO (spillai): We assume that the first input is the batched tensor
    first_input = list(cargs.values())[0]
    batch_size = first_input.shape[0]

    logger.debug(f"Tracing {slug} with NOS")
    try:
        concrete_args = concrete_args or {}
        tracer = options.tracer_cls()
        traced_graph = tracer.trace(model, concrete_args=concrete_args, dummy_inputs=cargs)
        traced_model = torch.fx.GraphModule(model, traced_graph)
        if options.verbose:
            logger.debug(">" * 80)
            logger.debug(f"Traced {slug}")
            with io.StringIO() as buf, contextlib.redirect_stdout(buf):
                traced_graph.print_tabular()
                logger.debug(buf.getvalue())
            logger.debug(">" * 80)
    except Exception as e:
        logger.error(f"Failed to trace model: {e}")
        logger.error(traceback.format_exc())
        return None
    logger.debug(f"Traced {slug} in {time.time() - st:.2f}s")

    st = time.time()
    logger.debug(f"Compiling {slug} with NOS")
    device_id = torch.cuda.current_device()
    free, total = torch.cuda.mem_get_info(device_id)
    log_mem_usage()

    max_workspace_size = free // 2
    logger.debug(f"Compiling with max_workspace_size = {max_workspace_size / 1024 ** 3:.1f} GiB")
    try:
        with io.StringIO() as buf, contextlib.redirect_stdout(buf):
            trt_model = torch_tensorrt.fx.compile(
                traced_model,
                list(cargs.values()),
                min_acc_module_size=5,
                max_workspace_size=max_workspace_size,
                explicit_batch_dimension=True,
                lower_precision=LowerPrecision.FP32 if precision == torch.float32 else LowerPrecision.FP16,
                verbose_log=options.verbose,
                timing_cache_prefix="",
                save_timing_cache=False,
                cuda_graph_batch_size=batch_size,
                dynamic_batch=False,
                max_batch_size=2048,
                is_aten=False,
                use_experimental_fx_rt=False,
            )
            logger.debug(buf.getvalue())
        assert callable(trt_model), "Compiled model must be callable"
    except Exception as e:
        logger.error(f"Failed to compile {slug}: {e}, skipping compilation")
        logger.error(traceback.format_exc())
        return None
    logger.debug(f"Compiled {slug} in {time.time() - st:.2f}s")

    # Finally, we use Torch utilities to clean up the workspace
    logger.debug("Cleaning up workspace")
    log_mem_usage()
    torch._dynamo.reset()
    with torch.no_grad():
        torch.cuda.empty_cache()
    del model

    if options.return_intermediates:
        logger.warning(
            f"Returning intermediate compilation results for {slug}, remember to cleanup traced models manually"
        )
        log_mem_usage()
        return NOSCompilerOutput(
            args=cargs,
            concrete_args=concrete_args,
            precision=precision,
            options=options,
            traced_model=traced_model,
            compiled_model=trt_model,
        )

    del traced_model
    del traced_graph
    return trt_model
