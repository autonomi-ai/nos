def test_noop():
    import numpy as np
    from PIL import Image

    from nos import hub
    from nos.common import TaskType
    from nos.test.utils import NOS_TEST_IMAGE

    noop = hub.load("noop/process-images", task=TaskType.CUSTOM)
    assert noop is not None

    img = Image.open(NOS_TEST_IMAGE)
    inputs = {"images": np.asarray(img)}
    outputs = noop.process_images(**inputs)
    assert outputs is not None
