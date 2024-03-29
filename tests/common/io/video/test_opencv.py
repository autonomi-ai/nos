import numpy as np
import pytest
from loguru import logger

from nos.common.io import VideoReader
from nos.test.utils import NOS_TEST_VIDEO


VIDEO_FILES = [NOS_TEST_VIDEO]


@pytest.mark.parametrize("filename", VIDEO_FILES)
def test_video_reader(filename):
    """Test VideoReader."""
    from itertools import islice

    logger.info(f"Testing video {filename}")
    video = VideoReader(filename)

    # Test len()
    assert len(video) > 0

    # Test __getitem__()
    indices = np.random.choice(np.arange(len(video)), size=min(5, len(video)), replace=False)
    for index in indices:
        img = video[index]
        assert isinstance(img, np.ndarray)

    # Test __getitem__() with np.int
    imgs = video[indices]
    assert isinstance(imgs, list)

    # Test __getitem__() with list
    imgs = video[indices.tolist()]
    assert isinstance(imgs, list)

    # Reset
    video.reset()
    assert video.pos() == 0

    # Test iterator
    for img in islice(video, 0, 10):
        assert isinstance(img, np.ndarray)

    # Seek to a random position and test pos()
    video.seek(1)
    assert video.pos() == 1

    # Test invalid file
    with pytest.raises(FileNotFoundError):
        VideoReader("does_not_exist.mp4")

    # Test invalid bridge
    with pytest.raises(NotImplementedError):
        VideoReader(filename, bridge="invalid_bridge")

    # Test out-of-bounds seek
    with pytest.raises(IndexError):
        video.seek(len(video) + 1)

    # Test __getitem__() with invalid types
    with pytest.raises(TypeError):
        video["invalid_index"]

    # Test __getitem__() with invalid index
    with pytest.raises(IndexError):
        video[len(video) + 1]

    # Test next() StopIteration after seeking to the end
    video.seek(len(video) - 1)
    img = next(video)
    with pytest.raises(StopIteration):
        next(video)


@pytest.mark.skip(reason="Not implemented")
def test_video_context_manager():
    """Test VideoReader context manager."""
    with VideoReader(NOS_TEST_VIDEO) as video:
        assert len(video) > 0
        img = next(video)
        assert isinstance(img, np.ndarray)


@pytest.mark.parametrize("filename", VIDEO_FILES)
def test_video_bridge(filename):
    """Test VideoReader bridge."""
    for (bridge, instance_type) in [
        ("numpy", np.ndarray),
    ]:
        video = VideoReader(filename, bridge=bridge)
        assert len(video) > 0
        img = next(video)
        assert isinstance(img, instance_type)


def test_video_writer():
    """Test VideoWriter."""
    import tempfile
    import uuid
    from pathlib import Path

    from nos.common.io import VideoReader, VideoWriter

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = str(Path(tmp_dir) / f"{str(uuid.uuid4().hex)}.avi")
        video = VideoReader(NOS_TEST_VIDEO, bridge="numpy")
        with VideoWriter(output_path) as writer:
            for img in video:
                writer.write(img)
        assert Path(output_path).exists()
        assert len(VideoReader(output_path)) == len(video)


@pytest.mark.benchmark
def test_benchmark_video_loading():
    """Benchmark VideoReader loading."""
    import time

    import requests
    from tqdm import tqdm

    from nos.constants import NOS_CACHE_DIR

    URL = "https://zackakil.github.io/video-intelligence-api-visualiser/assets/test_video.mp4"

    # Download video from URL to local file
    tmp_videos_dir = NOS_CACHE_DIR / "test_data" / "videos"
    tmp_videos_dir.mkdir(parents=True, exist_ok=True)
    tmp_video_filename = tmp_videos_dir / "test_video.mp4"
    if not tmp_video_filename.exists():
        with open(str(tmp_video_filename), "wb") as f:
            f.write(requests.get(URL).content)
        assert tmp_video_filename.exists()

    # Test video loading
    video = VideoReader(tmp_video_filename)
    st = time.perf_counter()
    for img in tqdm(video):
        assert isinstance(img, np.ndarray)
    end = time.perf_counter()
    logger.info(
        f"VideoReader:: nframes={len(video)}, shape={img.shape}, elapsed={end - st:.2f}s, fps={len(video) / (end-st):.1f}fps"
    )
