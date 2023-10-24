"""Pixeltable integration tests

Requirements:
- test in client-only environment (pip install torch-nos pixeltable)

Benchmarks:
- See benchmark-pixeltable.md
"""

from datetime import datetime
from pathlib import Path

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
                logger.debug(f"Stopping container: {container.name}")
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

    # Try to import pixeltable
    import pixeltable  # noqa: F401


BENCHMARK_IMAGE_SHAPES = [(640, 480), (1280, 720), (2880, 1620)]


def pixeltable_integration(write_profile: bool = False):
    import pandas as pd
    import pixeltable as pt

    from nos.common import timer
    from nos.common.io import VideoReader
    from nos.constants import NOS_CACHE_DIR
    from nos.test.utils import NOS_TEST_VIDEO, get_benchmark_video  # noqa: F401
    from nos.version import __version__

    pd.set_option("display.max_rows", 1000)
    pd.set_option("display.max_columns", 1000)

    NOS_INTEGRATIONS_DIR = Path(NOS_CACHE_DIR) / "integrations"
    NOS_INTEGRATIONS_DIR.mkdir(exist_ok=True, parents=True)

    # Get benchmark video, and read first frame for image dimensions
    FILENAME = str(NOS_TEST_VIDEO)
    # FILENAME = str(get_benchmark_video())

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
    import pixeltable.functions.image_generation as imagen  # noqa: F401
    from pixeltable.functions.custom import noop_process_images as noop
    from pixeltable.functions.image_embedding import openai_clip
    from pixeltable.functions.object_detection_2d import yolox_medium, yolox_tiny

    sdv21 = imagen.stabilityai_stable_diffusion_2_1  # noqa: F841
    sdxl = imagen.stabilityai_stable_diffusion_xl_base_1_0  # noqa: F841

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
    cl.drop_table("test_prompts", ignore_errors=True)

    PROMPTS = [["cat on a sofa", "blurry image"], ["astronaut on the moon, 4k, hdr", "low resolution"]]
    try:
        t = cl.get_table("test_data")
        prompts_t = cl.get_table("test_prompts")
    except Exception:
        t = cl.create_table(
            "test_data",
            cols,
            extract_frames_from="video",
            extracted_frame_col="frame",
            extracted_frame_idx_col="frame_idx",
            extracted_fps=0,
        )
        prompts_t = cl.create_table(
            "test_prompts", [pt.Column("prompt", pt.StringType()), pt.Column("neg_prompt", pt.StringType())]
        )
        prompts_t.insert(PROMPTS)

    # Resized columns
    # RH, RW = 480, 640
    for (RW, RH) in [(224, 224)] + BENCHMARK_IMAGE_SHAPES:
        t.add_column(pt.Column(f"frame_{RW}x{RH}", computed_with=t.frame.resize((RW, RH))))
    t.insert([VIDEO_FILES],columns=["video",],)  # fmt: skip

    # Run inference (see acceptance criteria from timing table above)
    timing_records = []

    # NOOP
    t[noop(t.frame)].show(1)  # noop (warmup)
    with timer(f"noop_{W}x{H}", n=nframes) as info:
        t.add_column(pt.Column("noop_ids", computed_with=noop(t.frame)))
    logger.debug(info)
    timing_records.append(info)

    for (RW, RH) in BENCHMARK_IMAGE_SHAPES:
        with timer(f"noop_{RW}x{RH}", n=nframes) as info:
            t.add_column(pt.Column(f"noop_ids_{RW}x{RH}", computed_with=noop(getattr(t, f"frame_{RW}x{RH}"))))
        logger.debug(info)
        timing_records.append(info)

    # YOLOX
    for (name, yolo_model) in [("medium", yolox_medium), ("tiny", yolox_tiny)]:
        t[yolo_model(t.frame)].show(1)  # load model
        with timer(f"yolox_{name}_{W}x{H}", n=nframes) as info:
            t.add_column(pt.Column(f"detections_yolo_{name}", computed_with=yolo_model(t.frame)))
        logger.debug(info)
        timing_records.append(info)

        for (RW, RH) in BENCHMARK_IMAGE_SHAPES:
            with timer(f"yolox_{name}_{RW}x{RH}", n=nframes) as info:
                t.add_column(
                    pt.Column(
                        f"detections_yolo_{name}_{RW}x{RH}", computed_with=yolo_model(getattr(t, f"frame_{RW}x{RH}"))
                    )
                )
            logger.debug(info)
            timing_records.append(info)

    # CLIP
    t[openai_clip(t.frame_224x224)].show(1)  # load model
    for (RW, RH) in [(224, 224)]:
        with timer(f"openai_{RW}x{RH}", n=nframes) as info:
            t.add_column(
                pt.Column(f"embedding_clip_{RW}x{RH}", computed_with=openai_clip(getattr(t, f"frame_{RW}x{RH}")))
            )
        logger.debug(info)
        timing_records.append(info)

    # SDv2
    H, W = 512, 512
    prompts_t[sdv21(prompts_t.prompt, prompts_t.neg_prompt, 1, 50, 7.5, H, W, 1)].show(1)  # load model
    with timer(f"sdv21_{W}x{H}", n=len(PROMPTS)) as info:
        prompts_t.add_column(
            pt.Column(
                "img_sdv21",
                computed_with=sdv21(
                    prompts_t.prompt,
                    prompts_t.neg_prompt,  # negative_prompts
                    1,  # num_images
                    50,  # num_inference_steps
                    7.5,  # guidance_scale
                    H,  # height
                    W,  # width
                    1,
                ),  # seed
                stored=True,
            )
        )
    logger.debug(info)
    timing_records.append(info)

    # SDXL
    H, W = 1024, 1024
    prompts_t[sdxl(prompts_t.prompt, prompts_t.neg_prompt, 1, 50, 7.5, H, W, 1)].show(1)  # load model
    with timer(f"sdxl_{W}x{H}", n=len(PROMPTS)) as info:
        prompts_t.add_column(
            pt.Column(
                "img_sdxl",
                computed_with=sdxl(prompts_t.prompt, prompts_t.neg_prompt, 1, 50, 7.5, H, W, 1),
                stored=True,
            )
        )
    logger.debug(info)
    timing_records.append(info)

    timing_df = pd.DataFrame([r.to_dict() for r in timing_records], columns=["desc", "elapsed", "n"])
    timing_df = timing_df.assign(
        elapsed=lambda x: x.elapsed.round(2),
        latency_ms=lambda x: ((x.elapsed / x.n) * 1000).round(2),
        fps=lambda x: (1 / (x.elapsed / x.n)).round(2),
    )
    logger.info(f"\nTiming records\n{timing_df}")

    if write_profile:
        # Save timing records
        version_str = __version__.replace(".", "-")
        date_str = datetime.utcnow().strftime("%Y%m%d")
        profile_path = Path(NOS_INTEGRATIONS_DIR) / f"nos-pixeltable-profile--{version_str}--{date_str}.json"
        timing_df.to_json(str(profile_path), orient="records", indent=2)
        logger.info(f"Saved timing records to: {str(profile_path)}")


def test_pixeltable_integration():
    pixeltable_integration()


@pytest.mark.benchmark()
def test_stress_pixeltable_integration():
    """10-hour stress test pixeltable integration."""
    from nos.common import tqdm

    for _ in tqdm(duration=10 * 60 * 60, desc="Pixeltable integration/stress tests"):
        pixeltable_integration()
