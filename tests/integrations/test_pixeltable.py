"""Pixeltable integration tests

Requirements:
- test in client-only environment (pip install autonomi-nos pixeltable)

Benchmarks:
- compute-with for noop, yolox/medium, openai/clip

"""
import contextlib
import time
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from loguru import logger


# Ensure that the environment does not have server-side requirements installed
try:
    import ray  # noqa: F401
    import torch  # noqa: F401

    raise AssertionError("torch/ray is installed in pixeltable environment")
except ImportError:
    pass

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


@dataclass
class TimingInfo:
    desc: str
    elapsed: float = field(init=False, default=None)

    def __repr__(self):
        return f"{self.__class__.__name__}(desc={self.desc}, elapsed={self.elapsed:.2f}s)"


@contextlib.contextmanager
def timer(desc: str = ""):
    info = TimingInfo(desc)

    logger.info(f"timer: {desc}")
    start = time.time()
    yield info
    info.elapsed = time.time() - start


def test_pixeltable_integration():
    import pixeltable as pt

    import nos
    from nos.common.io import VideoReader
    from nos.test.utils import get_benchmark_video

    print(nos.__version__)

    # Get benchmark video, and read first frame for image dimensions
    FILENAME = get_benchmark_video()
    assert Path(FILENAME).exists()
    _video = VideoReader(FILENAME)
    assert len(_video) > 0
    H, W, C = _video[0].shape

    # Setup videos to insert
    VIDEO_FILES = [FILENAME]

    # Force remove an existing pixeltable-store container
    # cleanup()

    # Import pixeltable client
    cl = pt.Client()

    # Import pixeltable functions (only available after client is initialized)
    from pixeltable.functions.custom import noop_process_images as noop
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

    # Compute detections
    records = []
    with timer(f"noop_{H}x{W}") as info:
        t.add_column(pt.Column("noop_ids", computed_with=noop(t.frame)))
    logger.info(info)
    records.append(info)
    # assert info.elapsed <= 200., f"noop on full resolution took too long, timing={info}"
    with timer("noop_640x480") as info:
        t.add_column(pt.Column("noop_ids_s", computed_with=noop(t.frame_s)))
    logger.info(info)
    records.append(info)
    # assert info.elapsed <= 130., f"noop on low-resolution took too long, timing={info}"
    with timer("yolox_medium_{H}x{W}") as info:
        t.add_column(pt.Column("detections_ym", computed_with=yolox_medium(t.frame)))
    logger.info(info)
    records.append(info)
    # assert info.elapsed <= 130., f"noop on low-resolution took too long, timing={info}"
    with timer("yolox_medium on resized") as info:
        t.add_column(pt.Column("detections_ym_s", computed_with=yolox_medium(t.frame_s)))
    logger.info(info)
    records.append(info)
    # assert info.elapsed <= 130., f"noop on low-resolution took too long, timing={info}"
