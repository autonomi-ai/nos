"""Test and benchmark YOLOX compilation with TorchTRT.

On NVIDIA 4090:

On (1x3x640x480):
 - torch-eager | yolox/medium: 1867it [00:20,  93.33it/s]
 - torch-eager | yolox/large:  1500it [00:20, 74.97it/s]

 -   torch-trt | yolox/tiny:   3722it [00:20, 186.08it/s]
 -   torch-trt | yolox/small:  3686it [00:20, 184.28it/s]
 -   torch-trt | yolox/medium: 3352it [00:20, 167.57it/s]
 -   torch-trt | yolox/large:  2617it [00:20, 130.83it/s]
 -   torch-trt | yolox/xlarge: 1812it [00:20, 90.57it/s]

On (1x3x1280x960):
-    torch-trt | yolox/tiny:   2052it [00:20, 102.58it/s]
-    torch-trt | yolox/small:  1990it [00:20, 99.50it/s]
-    torch-trt | yolox/medium: 1480it [00:20, 73.96it/s]
-    torch-trt | yolox/large:  1056it [00:20, 52.80it/s]
-    torch-trt | yolox/xlarge:  682it [00:20, 34.05it/s]


 Currently yolox/nano does not compile due to accuracy issues.
    AssertionError: Pass <function Lowerer.__call__.<locals>.do_lower at 0x7fdaf2e51af0> failed correctness check due at output 1:
    Tensor-likes are not close!

    Mismatched elements: 17 / 153600 (0.0%)
    Greatest absolute difference: 0.21282958984375 at index (0, 82, 1, 32) (up to 0.1 allowed)
    Greatest relative difference: 10.422420572267951 at index (0, 34, 23, 4) (up to 0.1 allowed)
"""
import os

import pytest

from nos.common import tqdm
from nos.logging import logger
from nos.models.yolox import YOLOX
from nos.test.utils import NOS_TEST_IMAGE, PyTestGroup


env = os.environ.get("NOS_ENV", os.getenv("CONDA_DEFAULT_ENV", "base_gpu"))
logger.info(f"Using env: {env}")
pytestmark = pytest.mark.skipif(
    env not in ("nos_trt_dev", "nos_trt_runtime"),
    reason=f"Requires nos env [nos_trt_dev, nos_trt_runtime], but using {env}",
)


# @pytest.mark.parametrize("precision", [torch.float32, torch.float16])
@pytest.mark.parametrize("shape", [(640, 480), (1280, 960)])
@pytest.mark.parametrize("model_name", list(YOLOX.configs.keys()))
@pytest.mark.benchmark(group=PyTestGroup.MODEL_COMPILATION)
def test_yolox_torchtrt_compilation(model_name, shape):
    """Test and benchmark compilation of YOLOX with TorchTRT."""
    from PIL import Image

    from nos.models.yolox import YOLOXTensorRT

    det = YOLOXTensorRT(model_name=model_name)
    img1 = Image.open(NOS_TEST_IMAGE)
    img1 = img1.resize(shape)

    # First run will trigger a compilation (if not already cached)
    predictions = det([img1])
    assert "bboxes" in predictions
    assert "scores" in predictions
    assert "labels" in predictions
    assert len(predictions["bboxes"]) == len(predictions["scores"]) == len(predictions["labels"])

    # Attempt running with a different image size
    with pytest.raises(Exception):
        img2 = img1.resize((320, 240))
        predictions = det([img2])

    # Subsequent runs will use the cached compilation
    logger.debug("Warming up (5s) ...")
    for _ in tqdm(duration=5.0, disable=True):
        predictions = det([img1])
    logger.debug("Running benchmark (20s) ...")
    for _ in tqdm(duration=20.0, desc=f"torch-trt | {model_name}"):
        predictions = det([img1])
