import contextlib
import time
from typing import Any

from tqdm import tqdm as _tqdm

from .cloudpickle import dumps, loads
from .shm import SharedMemoryDataDict, SharedMemoryNumpyObject, SharedMemoryTransportManager  # noqa: F401
from .spec import FunctionSignature, ModelSpec, ObjectTypeInfo
from .tasks import TaskType
from .types import Batch, EmbeddingSpec, ImageSpec, ImageT, TensorSpec, TensorT


def tqdm(iterable: Any = None, *args, **kwargs) -> Any:
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
            elapsed_ms = now_ms - st_ms
            if elapsed_ms >= duration_ms:
                return
            yield idx
            idx += 1

    return _tqdm(_iterable(), *args, **kwargs)


class TimingInfo:
    """Timing information for a context manager"""

    def __init__(self, desc: str, elapsed: float = 0.0, **kwargs):
        self.desc = desc
        self.elapsed = elapsed
        self.kwargs = kwargs

    def __repr__(self):
        repr_str = f"{self.__class__.__name__}(desc={self.desc}"
        if len(self.kwargs):
            repr_str += ", " + ", ".join([f"{k}={v}" for k, v in self.kwargs.items()])
        repr_str += f", elapsed={self.elapsed:.2f}s)"
        return repr_str

    def to_dict(self):
        return {**self.kwargs, "desc": self.desc, "elapsed": self.elapsed}


@contextlib.contextmanager
def timer(desc: str = "", **kwargs):
    """Simple context manager for timing code blocks"""
    info = TimingInfo(desc, **kwargs)
    start = time.time()
    yield info
    info.elapsed = time.time() - start
