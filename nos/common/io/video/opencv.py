from pathlib import Path
from typing import Callable, Iterator, List, Optional, Union

import cv2
import numpy as np

from nos.common.io.video.base import BaseVideoReader
from nos.logging import logger


T = np.ndarray


class VideoReader(BaseVideoReader):
    """Video Reader with OpenCV backend."""

    def __init__(self, filename: Union[str, Path], transform: Optional[Callable] = None, bridge: str = "numpy"):
        """Initialize video reader.

        Args:
            filename (Union[str, Path]): The path to the video file.
            transform (Optional[Callable], optional): A function to apply to each frame. Defaults to None.
            bridge (str, optional): The bridge type to use. Defaults to "numpy". Options are ("numpy", "torch").
        Raises:
            FileNotFoundError: If the video file does not exist.
            NotImplementedError: If the bridge type is not supported.
        """
        super().__init__()
        self.filename = Path(str(filename))
        if not self.filename.exists():
            raise FileNotFoundError(f"{self.filename} does not exist")
        self.transform = transform
        # TODO (spillai): Add support for torch bridge
        if bridge not in ("numpy",):
            raise NotImplementedError(f"Unknown bridge type {bridge}")
        self.bridge = bridge
        self._video = self.open()

    def __len__(self) -> int:
        """Return the number of frames in the video.

        Returns:
            int: The number of frames in the video.
        """
        return int(self._video.get(cv2.CAP_PROP_FRAME_COUNT))

    def __iter__(self) -> Iterator[T]:
        """Return an iterator over the video.

        Returns:
            Iterator[T]: An iterator over the video.
        """
        return self

    def __next__(self) -> T:
        """Return the next frame in the video.

        Raises:
            StopIteration: If there are no more frames in the video.
        Returns:
            T: The next frame in the video.
        """
        ret, img = self._video.read()
        if ret is False:
            raise StopIteration()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(img)
        return img

    def __getitem__(self, idx: Union[T, List[int], int]) -> Union[T, List[T]]:
        """Return the frame or frame(s) at the given index/indices.

        Args:
            idx (Union[T, List[int], int]): The index or indices to return.
        Raises:
            IndexError: If the index is out of bounds.
        Returns:
            Union[T, List[T]]: The frame or frame(s) at the given index/indices.
        """
        if isinstance(idx, (int, np.int32, np.int64)):
            self.seek(idx)
            return next(self)
        elif isinstance(idx, (T, List)):
            return [self.__getitem__(ind) for ind in idx]
        else:
            raise TypeError(f"Unknown type for {idx}, type={type(idx)}")

    def open(self) -> cv2.VideoCapture:
        """Open the video file.

        Raises:
            RuntimeError: If the video file cannot be opened.
        Returns:
            cv2.VideoCapture: The opened video file.
        """
        video = cv2.VideoCapture(str(self.filename))
        if not video.isOpened():
            raise RuntimeError(f"{self.__class__.__name__} :: Failed to open {self.filename}")
        return video

    def close(self) -> None:
        """Close the video file."""
        self._video.release()
        self._video = None

    def pos(self) -> Optional[int]:
        """Return the current position in the video.

        Returns:
            Optional[int]: The current position in the video.
        """
        try:
            return int(self._video.get(cv2.CAP_PROP_POS_FRAMES))
        except Exception:
            return None

    def seek(self, idx: int) -> None:
        """Seek to the given index in the video.

        Args:
            idx (int): The index to seek to.
        Raises:
            IndexError: If the index is out of bounds.
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Invalid index {idx}")
        self._video.set(cv2.CAP_PROP_POS_FRAMES, idx)

    def reset(self) -> None:
        """Reset the video to the beginning."""
        self.seek(0)


class VideoWriter:
    """Video Writer with OpenCV backend."""

    def __init__(self, filename: Union[str, Path], fps: int = 30):
        """Initialize video writer.

        Args:
            filename (Union[str, Path]): The path to the video file.
            fps (int, optional): The frames per second. Defaults to 30.
        """
        self.filename = Path(str(filename))
        if self.filename.exists():
            raise FileExistsError(f"{self.filename} already exists")
        self.fps = fps
        self.writer = None

    def write(self, img: np.ndarray) -> None:
        """Write the given image to the video file.

        Args:
            img (np.ndarray): The image to write.
        Returns:
            None
        """
        # Create video writer if it does not exist.
        if self.writer is None:
            H, W = img.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(str(self.filename), fourcc, self.fps, (W, H), img.ndim == 3)
            logger.debug(
                f"""{self.__class__.__name__} :: Create video writer """
                f"""[filename={self.filename}, W={W}, H={H}]"""
            )
        # Write image to video writer in BGR format.
        self.writer.write(img[..., ::-1])

    def close(self, reencode: bool = True):
        """Close the video writer, re-encoding the video if needed."""
        if self.writer is None:
            return
        logger.debug(f"Closing video writer [filename={self.filename}].")
        self.writer.release()
        self.writer = None
        # Re-encode video on releasing for better compatibility and
        # compression ratios.
        if reencode:
            try:
                VideoWriter.reencode_video(self.filename)
            except Exception as e:
                logger.warning(f"Failed to re-encode video, skipping [filename={self.filename}]: {e}")
        logger.debug(f"Closed video writer [filename={self.filename}].")

    def __del__(self):
        """Close the video writer cleanly."""
        self.close()

    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit context manager."""
        self.close()

    @staticmethod
    def reencode_video(filename: Union[str, Path]) -> str:
        """Re-encode video to ensure compatibility with OpenCV.

        Args:
            filename (Union[str, Path]): The path to the video file.

        Raises:
            FileNotFoundError: If the video file does not exist.

        Returns:
            str: The path to the re-encoded video file.
        """
        import uuid
        from pathlib import Path
        from subprocess import call

        input_path = Path(str(filename))
        if not input_path.exists():
            raise FileNotFoundError(f"{input_path} does not exist")
        output_path = input_path.parent / f"{str(uuid.uuid4().hex)}.mp4"
        try:
            logger.debug(f"re-encode video [filename={str(input_path)}, output={str(output_path)}]")
            cmd = (
                f"""ffmpeg -loglevel error -vsync 0 """
                f"""-i '{str(input_path)}' -c:v libx264 """
                f"""-pix_fmt yuv420p """
                f"""-vsync 0 -an {str(output_path)}"""
            )
            call([cmd], shell=True)
        except Exception:
            logger.error(f"re-encode video failed [filename={str(input_path)}]")
            logger.error(f"[cmd={cmd}]")
            raise

        if not output_path.exists():
            raise RuntimeError(f"Failed to re-encode video [filename={str(input_path)}]")
        logger.debug(
            f"""re-encode video overwriting [input={str(input_path)} """
            f"""({input_path.stat().st_size / 1024 ** 2:.2f} MB), """
            f"""output={str(output_path)} """
            f"""({output_path.stat().st_size / 1024 ** 2:.2f} MB)"""
        )
        logger.debug(f"re-encode video overwriting [filename={str(input_path)}]")
        output_path.rename(input_path)
        return str(input_path)
