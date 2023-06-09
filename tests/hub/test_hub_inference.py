from loguru import logger

from nos import hub
from nos.common import TaskType
from nos.test.utils import skip_if_no_torch_cuda


def hub_image_models():
    models = hub.list()
    assert len(models) > 0
    return [
        spec
        for spec in models
        if spec.task
        in (
            TaskType.IMAGE_EMBEDDING,
            TaskType.OBJECT_DETECTION_2D,
            TaskType.DEPTH_ESTIMATION_2D,
        )
        and not spec.name.endswith("-trt")
    ]


@skip_if_no_torch_cuda
def test_hub_batched_image_inference():
    import numpy as np
    from PIL import Image

    from nos.test.utils import NOS_TEST_IMAGE

    pil_im = Image.open(NOS_TEST_IMAGE)
    pil_im = pil_im.resize((640, 480))
    np_im = np.asarray(pil_im)

    # Test various batch types for each model
    # TOFIX (spillai): Currently all models don't support batched inference
    # with stacked np.ndarray.
    image_batch_types = [
        ("Image.Image", pil_im),
        ("np.ndarray", np_im),
        ("List[Image.Image]", [pil_im, pil_im]),
        ("List[np.ndarray]", [np_im, np_im]),
        # ("stacked np.ndarray", np.stack([np_im, np_im]))
    ]

    for spec in hub_image_models():
        logger.debug(f"Testing model [name={spec.name}, task={spec.task}]")

        # Run inference for each model (image-based models only)
        model = hub.load(spec.name, spec.task)
        predict = getattr(model, spec.signature.method_name)

        # Check if the model supports batched inference
        for input_type, images in image_batch_types:
            # Run inference
            try:
                predict(images=images)
            except Exception:
                import traceback

                traceback.print_exc()
                raise RuntimeError(
                    f"Model [name={spec.name}, task={spec.task}] does not support batched inference for input type '{input_type}'"
                )

        del model
