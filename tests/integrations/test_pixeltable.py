"""Pixeltable integration tests

Requirements:
- test in client-only environment (pip install autonomi-nos pixeltable)

Benchmarks:
- compute-with for noop, yolox/medium, openai/clip

Timing records (0.0.7 - 2023-07-14) w/o SHM
                      desc  elapsed    n  latency_ms    fps
0             noop_294x240     1.87  168       11.13  89.84
1             noop_640x480     1.88  168       11.19  89.36
2            noop_1280x720     5.97  168       35.54  28.14
3           noop_2880x1620    18.00  168      107.14   9.33
4     yolox_medium_294x240     2.98  168       17.74  56.38
5     yolox_medium_640x480     2.92  168       17.38  57.53
6    yolox_medium_1280x720    10.94  168       65.12  15.36
7   yolox_medium_2880x1620    23.02  168      137.02   7.30
8           openai_224x224     1.74  168       10.36  96.55
9           openai_640x480     3.28  168       19.52  51.22
10         openai_1280x720     7.84  168       46.67  21.43
11        openai_2880x1620    37.06  168      220.60   4.53

Timing records (0.0.7 - 2023-07-14) w/ SHM
                      desc  elapsed    n  latency_ms     fps
0             noop_294x240     1.15  168        6.85  146.09
1             noop_640x480     1.11  168        6.61  151.35
2            noop_1280x720     4.86  168       28.93   34.57
3           noop_2880x1620    16.83  168      100.18    9.98
4     yolox_medium_294x240     2.09  168       12.44   80.38
5     yolox_medium_640x480     1.97  168       11.73   85.28
6    yolox_medium_1280x720     9.12  168       54.29   18.42
7   yolox_medium_2880x1620    21.14  168      125.83    7.95
8           openai_224x224     1.34  168        7.98  125.37
9           openai_640x480     2.50  168       14.88   67.20
10         openai_1280x720     5.05  168       30.06   33.27
11        openai_2880x1620    18.72  168      111.43    8.97

Timing records (0.0.8 - 2023-08-01) w/ SHM
                      desc  elapsed    n  latency_ms     fps
0             noop_294x240     1.11  168        6.61  151.35
1             noop_640x480     1.08  168        6.43  155.56
2            noop_1280x720     4.82  168       28.69   34.85
3           noop_2880x1620    16.83  168      100.18    9.98
4     yolox_medium_294x240     2.05  168       12.20   81.95
5     yolox_medium_640x480     2.03  168       12.08   82.76
6    yolox_medium_1280x720     9.23  168       54.94   18.20
7   yolox_medium_2880x1620    21.26  168      126.55    7.90
8           openai_224x224     1.29  168        7.68  130.23
9           openai_640x480     2.51  168       14.94   66.93
10         openai_1280x720     4.94  168       29.40   34.01
11        openai_2880x1620    18.49  168      110.06    9.09

"""

from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest
from loguru import logger


# Skip this entire test if pixeltable is not installed
pytestmark = pytest.mark.skipif(pytest.importorskip("pixeltable") is None, reason="pixeltable is not installed")


PIXELTABLE_DB_NAME = "nos_test"
PIXELTABLE_CONTAINER_NAME = "pixeltable-store"


def cleanup():
    """Remove any existing containers with the pixeltable-store name"""
    import docker
    import nos

    # Cleanup pixeltable store container
    client = docker.from_env()
    for container in client.containers.list():
        if container.name == PIXELTABLE_CONTAINER_NAME:
            try:
                container.remove(force=True)
                logger.info(f"Stopping container: {container.name}")
            except Exception as e:
                raise RuntimeError(f"Failed to shutdown inference server: {e}")

    # Shutdown nos inference server
    nos.shutdown()


def test_pixeltable_installation():
    # Ensure that the environment does not have server-side requirements installed
    try:
        import ray  # noqa: F401
        import torch  # noqa: F401

        raise AssertionError("torch/ray is installed in pixeltable environment")
    except ImportError:
        pass


BENCHMARK_IMAGE_SHAPES = [(640, 480), (1280, 720), (2880, 1620)]


def test_pixeltable_integration():
    import pixeltable as pt

    from nos.common import timer
    from nos.common.io import VideoReader
    from nos.constants import NOS_CACHE_DIR
    from nos.test.utils import NOS_TEST_VIDEO, get_benchmark_video  # noqa: F401
    from nos.version import __version__

    NOS_INTEGRATIONS_DIR = Path(NOS_CACHE_DIR) / "integrations"
    NOS_INTEGRATIONS_DIR.mkdir(exist_ok=True, parents=True)

    # Get benchmark video, and read first frame for image dimensions
    FILENAME = str(NOS_TEST_VIDEO)
    # FILENAME = get_benchmark_video()

    assert Path(FILENAME).exists()
    _video = VideoReader(FILENAME)
    nframes = len(_video)
    assert nframes > 0
    H, W, C = _video[0].shape

    # Setup videos to insert
    VIDEO_FILES = [FILENAME]

    # Force remove an existing pixeltable-store container
    # cleanup()

    # Import pixeltable client
    cl = pt.Client()

    # Import pixeltable functions (only available after client is initialized)
    from pixeltable.functions.custom import noop_process_images as noop
    from pixeltable.functions.image_embedding import openai_clip
    from pixeltable.functions.object_detection_2d import yolox_medium

    # Setup pixeltable database
    try:
        cl.create_db(PIXELTABLE_DB_NAME)
    except Exception:
        pass

    # Setup columns
    cols = [
        pt.Column("video", pt.VideoType()),
        pt.Column("frame", pt.ImageType()),
        pt.Column("frame_idx", pt.IntType()),
    ]

    # Setup pixeltable test_data table
    cl.drop_table("test_data", ignore_errors=True)
    try:
        t = cl.get_table("test_data")
    except Exception:
        t = cl.create_table(
            "test_data",
            cols,
            extract_frames_from="video",
            extracted_frame_col="frame",
            extracted_frame_idx_col="frame_idx",
            extracted_fps=0,
        )

    # Resized columns
    # RH, RW = 480, 640
    for (RW, RH) in [(224, 224)] + BENCHMARK_IMAGE_SHAPES:
        t.add_column(pt.Column(f"frame_{RW}x{RH}", computed_with=t.frame.resize((RW, RH))))

    # Insert video files, and compute detections
    t.insert_rows(
        [
            [
                FILENAME,
            ]
            for path in VIDEO_FILES
        ],
        columns=[
            "video",
        ],
    )

    # Run inference (see acceptance criteria from timing table above)
    timing_records = []
    t[noop(t.frame)].show(1)  # noop (warmup)
    with timer(f"noop_{W}x{H}", n=nframes) as info:
        t.add_column(pt.Column("noop_ids", computed_with=noop(t.frame)))
    logger.info(info)
    timing_records.append(info)

    for (RW, RH) in BENCHMARK_IMAGE_SHAPES:
        with timer(f"noop_{RW}x{RH}", n=nframes) as info:
            t.add_column(pt.Column(f"noop_ids_{RW}x{RH}", computed_with=noop(getattr(t, f"frame_{RW}x{RH}"))))
        logger.info(info)
        timing_records.append(info)

    t[yolox_medium(t.frame)].show(1)  # load model
    with timer(f"yolox_medium_{W}x{H}", n=nframes) as info:
        t.add_column(pt.Column("detections_ym", computed_with=yolox_medium(t.frame)))
    logger.info(info)
    timing_records.append(info)

    for (RW, RH) in BENCHMARK_IMAGE_SHAPES:
        with timer(f"yolox_medium_{RW}x{RH}", n=nframes) as info:
            t.add_column(
                pt.Column(f"detections_ym_{RW}x{RH}", computed_with=yolox_medium(getattr(t, f"frame_{RW}x{RH}")))
            )
        logger.info(info)
        timing_records.append(info)

    t[openai_clip(t.frame)].show(1)  # load model
    for (RW, RH) in [(224, 224)] + BENCHMARK_IMAGE_SHAPES:
        with timer(f"openai_{RW}x{RH}", n=nframes) as info:
            t.add_column(
                pt.Column(f"embedding_clip_{RW}x{RH}", computed_with=openai_clip(getattr(t, f"frame_{RW}x{RH}")))
            )
        logger.info(info)
        timing_records.append(info)

    timing_df = pd.DataFrame([r.to_dict() for r in timing_records], columns=["desc", "elapsed", "n"])
    timing_df = timing_df.assign(
        elapsed=lambda x: x.elapsed.round(2),
        latency_ms=lambda x: ((x.elapsed / nframes) * 1000).round(2),
        fps=lambda x: (1 / (x.elapsed / nframes)).round(2),
    )
    logger.info(f"\nTiming records\n{timing_df}")

    # Save timing records
    version_str = __version__.replace(".", "-")
    date_str = datetime.utcnow().strftime("%Y%m%d")
    profile_path = Path(NOS_INTEGRATIONS_DIR) / f"nos-pixeltable-profile--{version_str}--{date_str}.json"
    timing_df.to_json(str(profile_path), orient="records", indent=2)
    logger.info(f"Saved timing records to: {str(profile_path)}")
