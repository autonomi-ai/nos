"""Test and benchmarking CLIP compilation with TorchTRT.

On NVIDIA 4090:
 - [openai/clip-vit-base-patch32] Eager: 5707it [00:30, 190.22it/s]
 - [openai/clip-vit-base-patch32] TRT: 23968it [00:30, 798.92it/s]
"""
import os

import pytest

from nos.common import tqdm
from nos.constants import NOS_TMP_DIR
from nos.logging import logger
from nos.test.utils import PyTestGroup


env = os.environ.get("NOS_ENV", os.getenv("CONDA_DEFAULT_ENV", "base_gpu"))
logger.info(f"Using env: {env}")
pytestmark = pytest.mark.skipif(
    env not in ("nos_trt_dev", "nos_trt_runtime"),
    reason=f"Requires nos env [nos_trt_dev, nos_trt_runtime], but using {env}",
)


@pytest.mark.benchmark(group=PyTestGroup.MODEL_COMPILATION)
def test_clip_torchtrt_compilation():
    """Test and benchmark compilation of CLIP with TorchTRT."""

    import torch
    import torch_tensorrt.fx.converter_registry as registry
    import torch_tensorrt.fx.tracer.acc_tracer.acc_tracer as acc_tracer
    from torch_tensorrt.fx import InputTensorSpec, TRTInterpreter, TRTModule
    from torch_tensorrt.fx.tools.trt_splitter import TRTSplitter

    # De-register certain ops from TRT
    from torch_tensorrt.fx.tracer.acc_tracer import acc_ops
    from torch_tensorrt.fx.utils import LowerPrecision
    from transformers import CLIPVisionModel
    from transformers.modeling_outputs import BaseModelOutputWithPooling
    from transformers.utils.fx import symbolic_trace

    from nos.compilers.trt import get_submod_inputs

    logger.warning(f"Deregistering acc_ops.expand: {acc_ops.expand in registry.CONVERTERS.keys()}")
    registry.CONVERTERS.pop(acc_ops.expand)
    logger.warning(f"Succesfully degistered acc_ops.expand: {acc_ops.expand not in registry.CONVERTERS.keys()}")

    MODEL_NAME = "openai/clip-vit-base-patch32"
    model = CLIPVisionModel.from_pretrained(MODEL_NAME).cuda().eval()
    inputs = [torch.randn((1, 3, 224, 224), dtype=torch.float32, device="cuda")]

    tmp_dir = NOS_TMP_DIR / "models"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_filename = tmp_dir / "clip_trt_fp32.pth"
    if not tmp_filename.exists():
        # Symbolically trace model with HFTracer
        logger.debug("Tracing model with HFTracer")
        with torch.inference_mode():
            traced = symbolic_trace(
                model,
                input_names=["pixel_values"],
                disable_check=False,
            )
            logger.debug(f"Traced model: {traced}")

        # Re-trace model with TRTTracer (normalizing kwargs to args)
        logger.debug("Tracing model with TRTTracer")
        with torch.inference_mode():
            trt_traced = acc_tracer.trace(
                traced,
                inputs,
            )
            logger.debug(f"TRTTraced model: {trt_traced}")

        # Splitter will split the model into several submodules. The name of submodules will
        # be either `run_on_acc_{}` or `run_on_gpu_{}`. Submodules named `run_on_acc_{}` can
        # be fully lowered to TensorRT via fx2trt while submodules named `run_on_gpu_{}` has
        # unsupported ops and can't be lowered by fx2trt. We can still run `run_on_gpu_{}`
        # submodules on Gpu if ops there have cuda implementation, the naming is a bit
        # confusing and we'll improve it.
        logger.info("Splitting model with TRTSplitter")
        splitter = TRTSplitter(trt_traced, inputs)
        logger.debug(f"Splitted model: {splitter}")

        # Preview functionality allows us to see what are the supported ops and unsupported
        # ops. We can optionally the dot graph which will color supported ops and unsupported
        # ops differently.
        logger.debug("Previewing TRTSplitter")
        _ = splitter.node_support_preview(dump_graph=False)

        # Callable that will be used in place of nn.Module that contains the
        # splitted accelerated and non-accelerated ops.
        split_mod = splitter()

        # Since the model is splitted into three segments. We need to lower each TRT eligible segment.
        # If we know the model can be fully lowered, we can skip the splitter part.
        for name, _ in split_mod.named_children():
            logger.debug(f"Splitting {name}")
            if "_run_on_acc" in name:
                submod = getattr(split_mod, name)

                # Get submodule inputs for fx2trt
                acc_inputs = get_submod_inputs(split_mod, submod, inputs)

                # fx2trt replacement
                interp = TRTInterpreter(
                    submod,
                    InputTensorSpec.from_tensors(acc_inputs),
                    explicit_batch_dimension=True,
                )
                r = interp.run(lower_precision=LowerPrecision.FP32)
                trt_mod = TRTModule(*r)
                setattr(split_mod, name, trt_mod)

        # Save the model
        logger.debug("Saving model")

        torch.save(split_mod, str(tmp_filename))
        logger.debug(f"Saved model to {tmp_filename}")
        del split_mod
        del splitter
        del trt_traced
        del traced

    # Load the model
    logger.debug("Loading model")
    trt_model = torch.load(str(tmp_filename))
    logger.debug(f"Loaded model: {trt_model}")

    # Compare outputs keys
    with torch.inference_mode():
        trt_output = trt_model(*inputs)
        # Check if we can rebuild the same model output as CLIPVisionModel
        _trt_output = BaseModelOutputWithPooling(**trt_output)
    with torch.inference_mode():
        eager_output = model(*inputs)
        # Note: Convert eager output to dict to match trt output
        eager_output = dict(eager_output)
    assert trt_output.keys() == eager_output.keys()
    for key in trt_output.keys():
        torch.testing.assert_close(
            trt_output[key].cpu().float(),
            eager_output[key].cpu().float(),
            rtol=2e-02,
            atol=2e-02,
            equal_nan=True,
        )

    # Benchmark
    logger.debug("Benchmarking")
    with torch.inference_mode():
        for _ in tqdm(duration=30, desc=f"Benchmark [{MODEL_NAME}] Eager"):
            model(*inputs)

        for _ in tqdm(duration=30, desc=f"Benchmark [{MODEL_NAME}] TRT"):
            trt_model(*inputs)
