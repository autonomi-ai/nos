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
