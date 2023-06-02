from pathlib import Path
from typing import Callable, Iterator, List, Optional, Union

import cv2
import numpy as np

from nos.common.io.video.base import BaseVideoFile


T = np.ndarray


class VideoFile(BaseVideoFile):
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
