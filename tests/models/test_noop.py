def test_noop():
    import numpy as np
    from PIL import Image

    from nos import hub
    from nos.test.utils import NOS_TEST_IMAGE

    noop = hub.load("noop/process-images")
    assert noop is not None

    img = Image.open(NOS_TEST_IMAGE)
    inputs = {"images": np.asarray(img)}
    outputs = noop.process_images(**inputs)
    assert outputs is not None
    assert len(outputs) == 1

    inputs = {"images": np.stack([np.asarray(img) for _ in range(2)])}
    outputs = noop.process_images(**inputs)
    assert outputs is not None
    assert len(outputs) == 2

    inputs = {"images": img}
    outputs = noop.process_images(**inputs)
    assert outputs is not None
    assert len(outputs) == 1

    inputs = {"images": [img for _ in range(2)]}
    outputs = noop.process_images(**inputs)
    assert outputs is not None
    assert len(outputs) == 2
