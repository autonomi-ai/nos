"""Pixeltable integration tests

Requirements:
- test in client-only environment (pip install autonomi-nos pixeltable)

Benchmarks:
- compute-with for noop, yolox/medium, openai/clip

Timing records (2023-07-14)
                   desc  elapsed    n  latency_ms    fps
0          noop_294x240     4.54  168       27.02  37.00
1          noop_640x480     2.00  168       11.90  84.00
2  yolox_medium_294x240     3.11  168       18.51  54.02
3  yolox_medium_640x480     2.98  168       17.74  56.38
4        openai_640x480     3.33  168       19.82  50.45
"""
import contextlib
import time
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


class TimingInfo:
    def __init__(self, desc: str, **kwargs):
        self.desc = desc
        self.elapsed = None
        self.kwargs = kwargs

    def __repr__(self):
        repr_str = f"{self.__class__.__name__}(desc={self.desc}"
        if len(self.kwargs):
            repr_str += ", " + ", ".join([f"{k}={v}" for k, v in self.kwargs.items()])
        repr_str += f", elapsed={self.elapsed:.2f}s)"
        return repr_str

    def to_dict(self):
        return self.kwargs | {"desc": self.desc, "elapsed": self.elapsed}


@contextlib.contextmanager
def timer(desc: str = "", **kwargs):
    info = TimingInfo(desc, **kwargs)

    logger.info(f"timer: {desc}")
    start = time.time()
    yield info
    info.elapsed = time.time() - start


def test_pixeltable_installation():
    # Ensure that the environment does not have server-side requirements installed
    try:
        import ray  # noqa: F401
        import torch  # noqa: F401

        raise AssertionError("torch/ray is installed in pixeltable environment")
    except ImportError:
        pass


def test_pixeltable_integration():
    import pixeltable as pt

    import nos
    from nos.common.io import VideoReader
    from nos.constants import NOS_CACHE_DIR
    from nos.test.utils import NOS_TEST_VIDEO, get_benchmark_video  # noqa: F401

    NOS_INTEGRATIONS_DIR = Path(NOS_CACHE_DIR) / "integrations"
    NOS_INTEGRATIONS_DIR.mkdir(exist_ok=True, parents=True)

    print(nos.__version__)

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
        db = cl.create_db(PIXELTABLE_DB_NAME)
    except Exception:
        pass
    finally:
        db = cl.get_db(PIXELTABLE_DB_NAME)
    assert db is not None

    # Setup columns
    cols = [
        pt.Column("video", pt.VideoType(), nullable=False),
        pt.Column("frame", pt.ImageType(), nullable=False),
        pt.Column("frame_idx", pt.IntType(), nullable=False),
    ]

    # Setup pixeltable test_data table
    db.drop_table("test_data", ignore_errors=True)
    try:
        t = db.get_table("test_data")
    except Exception:
        t = db.create_table(
            "test_data",
            cols,
            extract_frames_from="video",
            extracted_frame_col="frame",
            extracted_frame_idx_col="frame_idx",
            extracted_fps=0,
        )

    # Resized columns
    RH, RW = 480, 640
    t.add_column(pt.Column("frame_s", computed_with=t.frame.resize((RW, RH))))

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
    with timer(f"noop_{W}x{H}", n=nframes) as info:
        t.add_column(pt.Column("noop_ids", computed_with=noop(t.frame)))
    logger.info(info)
    timing_records.append(info)
    assert info.elapsed <= 10.0, f"{info.desc} took too long, timing={info}"

    with timer(f"noop_{RW}x{RH}", n=nframes) as info:
        t.add_column(pt.Column("noop_ids_s", computed_with=noop(t.frame_s)))
    logger.info(info)
    timing_records.append(info)
    assert info.elapsed <= 4.0, f"{info.desc} took too long, timing={info}"

    t[yolox_medium(t.frame)].show(1)  # load model
    with timer(f"yolox_medium_{W}x{H}", n=nframes) as info:
        t.add_column(pt.Column("detections_ym", computed_with=yolox_medium(t.frame)))
    logger.info(info)
    timing_records.append(info)
    assert info.elapsed <= 5.0, f"{info.desc} took too long, timing={info}"

    with timer(f"yolox_medium_{RW}x{RH}", n=nframes) as info:
        t.add_column(pt.Column("detections_ym_s", computed_with=yolox_medium(t.frame_s)))
    logger.info(info)
    timing_records.append(info)
    assert info.elapsed <= 5.0, f"{info.desc} took too long, timing={info}"

    t[openai_clip(t.frame)].show(1)  # load model
    with timer(f"openai_{RW}x{RH}", n=nframes) as info:
        t.add_column(pt.Column("embedding_clip_s", computed_with=openai_clip(t.frame_s)))
    logger.info(info)
    timing_records.append(info)
    assert info.elapsed <= 5.0, f"{info.desc} took too long, timing={info}"

    timing_df = pd.DataFrame([r.to_dict() for r in timing_records], columns=["desc", "elapsed", "n"])
    timing_df = timing_df.assign(
        elapsed=lambda x: x.elapsed.round(2),
        latency_ms=lambda x: ((x.elapsed / nframes) * 1000).round(2),
        fps=lambda x: (1 / (x.elapsed / nframes)).round(2),
    )
    logger.info(f"\nTiming records\n{timing_df}")

    # Save timing records
    version_str = nos.__version__.replace(".", "-")
    date_str = datetime.utcnow().strftime("%Y%m%d")
    profile_path = Path(NOS_INTEGRATIONS_DIR) / f"nos-pixeltable-profile--{version_str}--{date_str}.json"
    timing_df.to_json(str(profile_path), orient="records", indent=2)
    logger.info(f"Saved timing records to: {str(profile_path)}")
