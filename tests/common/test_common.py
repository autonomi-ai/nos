from nos.common import tqdm


def test_common_tqdm():
    import time

    # Use tqdm as a regular progress bar
    for i1, i2 in zip(range(10), tqdm(range(10))):
        assert i1 == i2

    # Use tqdm as a timer
    for _ in tqdm(duration=1):
        time.sleep(0.1)
