import contextlib
import time
from typing import Any

import psutil
from tqdm import tqdm as _tqdm

from .cloudpickle import dumps, loads
from .runtime import RuntimeEnv
from .shm import SharedMemoryDataDict, SharedMemoryNumpyObject, SharedMemoryTransportManager  # noqa: F401
from .spec import (
    FunctionSignature,
    ModelDeploymentSpec,
    ModelResources,
    ModelServiceSpec,
    ModelSpec,
    ModelSpecMetadata,
    ModelSpecMetadataCatalog,
    ObjectTypeInfo,
)
from .tasks import TaskType
from .types import Batch, EmbeddingSpec, ImageSpec, ImageT, TensorSpec, TensorT


def tqdm(iterable: Any = None, *args, skip: int = 0, **kwargs) -> Any:
    """Wrapper around tqdm that allows for a duration to be
    specified instead of an iterable

    Args:
        iterable (Any, optional): Iterable to wrap. Defaults to None.
    Returns:
        tqdm.tqdm: Wrapped tqdm iterable.
    """
    if iterable is not None:
        return _tqdm(iterable, *args, **kwargs)

    # Get duration in milliseconds
    try:
        duration_s = kwargs.pop("duration")
        duration_ms = duration_s * 1_000
    except KeyError:
        raise KeyError("`duration` must be specified when no iterable is provided")

    # Yield progress bar for the specified duration
    def _iterable():
        idx = 0
        st_ms = time.perf_counter() * 1_000
        while True:
            now_ms = time.perf_counter() * 1_000
            if skip > 0 and idx < skip:
                st_ms = now_ms
            elapsed_ms = now_ms - st_ms
            if elapsed_ms >= duration_ms:
                return
            yield idx
            idx += 1

    return _tqdm(_iterable(), *args, **kwargs)


class TimingInfo(dict):
    """Timing information for a context manager"""

    def __init__(self, desc: str, elapsed: float = 0.0, **kwargs):
        """Initialize timing information

        Args:
            desc (str): Description of the context manager.
            elapsed (float, optional): Elapsed time. Defaults to 0.0.
            kwargs (Any): Additional key-value pairs to store.
        """
        self.__dict__ = self
        self.desc = desc
        self.elapsed = elapsed
        super().__init__(**kwargs)

    def __repr__(self):
        repr_str = f"{self.__class__.__name__}("
        if len(self):
            repr_str += ", ".join([f"{k}={v}" for k, v in self.items()])
        repr_str += ")"
        return repr_str

    def to_dict(self):
        return self.__dict__


@contextlib.contextmanager
def timer(desc: str = "", **kwargs):
    """Simple context manager for timing code blocks

    Args:
        desc (str, optional): Description of the context manager. Defaults to "".
        kwargs (Any): Additional key-value pairs to store.

    Yields:
        TimingInfo: Timing information with additional fields (elapsed time, cpu_util, etc.)
    """
    info = TimingInfo(desc, **kwargs)
    start = time.time()
    _ = psutil.cpu_percent(interval=None)
    yield info
    info.elapsed = round(time.time() - start, 3)
    info.cpu_util = round(psutil.cpu_percent(interval=None), 2)
