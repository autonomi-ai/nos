import os
from dataclasses import dataclass, field
from multiprocessing.managers import SharedMemoryManager
from multiprocessing.shared_memory import SharedMemory
from typing import Any, Dict, Tuple

import numpy as np

from nos.common.cloudpickle import dumps, loads
from nos.common.types import TensorSpec
from nos.logging import logger


NOS_SHM_ENABLED = bool(int(os.environ.get("NOS_SHM_ENABLED", 0)))


@dataclass
class SharedMemoryNumpyObject:
    """Shared memory object wrapping numpy array."""

    shm: SharedMemory
    """Shared memory object"""
    shape: Tuple[int, ...]
    """numpy array shape"""
    dtype: str
    """numpy array dtype"""

    def __repr__(self) -> str:
        """Return the shared memory object representation."""
        return f"SharedMemoryNumpyObject(name={self.name}, shape={self.shape}, dtype={self.dtype})"

    def __getstate__(self) -> Dict[str, Any]:
        """Return the shared memory object state."""
        return {"name": self.name, "shape": self.shape, "dtype": self.dtype}

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Set the shared memory object state."""
        self.__init__(SharedMemory(name=state["name"]), state["shape"], state["dtype"])

    def __del__(self) -> None:
        """Close the shared memory object."""
        self.shm.close()

    @property
    def name(self) -> str:
        """Return the shared memory name."""
        return self.shm.name

    def copy(self, arr: np.ndarray) -> None:
        """Copy the numpy array to shared memory object."""
        assert arr.shape == self.shape, f"Array shape {arr.shape} does not match shared memory shape {self.shape}"
        assert arr.dtype == self.dtype, f"Array dtype {arr.dtype} does not match shared memory dtype {self.dtype}"
        target = np.ndarray(arr.shape, dtype=arr.dtype, buffer=self.shm.buf)
        target[:] = arr[:]

    def get(self) -> np.ndarray:
        """Get the numpy array from the shared memory object ."""
        src = np.ndarray(shape=self.shape, dtype=getattr(np, self.dtype), buffer=self.shm.buf)
        target = src.copy()
        del src
        return target


class SharedMemoryDataDict:
    """Shared-memory data wrapper."""

    @staticmethod
    def decode(data: Dict[str, Any]) -> Dict[str, Any]:
        """Decode the data dictionary with shared-memory references.

        Note: First unpickle the data bytes, then replace the shared-memory references
        with numpy arrays. SharedMemoryNumpyObject have a custom __getstate__ method
        that returns the shared-memory name, shape and dtype. The __setstate__ method
        is called when the object is unpickled, and it creates a new SharedMemoryNumpyObject
        instance with the given name, shape and dtype.
        """
        data = {k: loads(v) for k, v in data.items()}
        if NOS_SHM_ENABLED:
            logger.debug("Decoding shared memory data")
            for k, v in data.items():
                if isinstance(v, SharedMemoryNumpyObject):
                    data[k] = v.get()
                    logger.debug(f"Decoded data: {k}, {type(data[k])}, {data[k].shape}")
                elif isinstance(v, list) and isinstance(v[0], SharedMemoryNumpyObject):
                    data[k] = [x.get() for x in v]
                    logger.debug(f"Decoded data: {k}, {type(data[k][0])}, {data[k][0].shape}")
        return data

    @staticmethod
    def encode(data: Dict[str, Any]) -> Dict[str, Any]:
        """Encode the data dictionary with shared-memory references."""
        return {k: dumps(v) for k, v in data.items()}


@dataclass
class SharedMemoryTransportManager:
    """Shared memory transport manager."""

    _manager: SharedMemoryManager = field(init=False)
    """Shared memory manager."""
    _objects_map: Dict[str, SharedMemory] = field(default_factory=dict)
    """Shared memory objects map."""

    def __post_init__(self):
        self._manager = SharedMemoryManager()
        self._manager.start()

    def __del__(self):
        """Clean up the shared memory."""
        self._manager.shutdown()

    def create(self, data: Dict[str, Any], namespace: str = "") -> Dict[str, Any]:
        """Create a shared memory segment for the data dictionary.

        Args:
            data (Dict[str, Any]): Inputs to the model ("images", "texts", "prompts" etc) as
                a dictionary of numpy arrays.
            namespace (str, optional): Unique namespace for the shared memory segment. Defaults to "".
        Returns:
            Dict[str, Any]: Shared memory segment for the data dictionary.
        """
        assert len(self._objects_map) == 0, "Shared memory segments should be empty."

        # Create shared memory segments for numpy arrays (or lists of numpy arrays)
        objects_map = {}
        for key, value in data.items():
            # Note (spillai): Assumes all items in the list are numpy arrays of the same shape.
            if isinstance(value, TensorSpec) or (isinstance(value, list) and isinstance(value[0], TensorSpec)):
                if isinstance(value, list):
                    nbytes = value[0].nbytes * len(value)
                    shape, dtype = value[0].shape, value[0].dtype
                    objects_map[key] = [
                        SharedMemoryNumpyObject(self._manager.SharedMemory(item.nbytes), item.shape, str(dtype))
                        for item in value
                    ]
                    logger.debug(
                        f"Created shm segment: key={f'{id}/{key}'}, size={nbytes / 1024 / 1024:.2f} MB, shape={shape}, dtype={dtype}, len={len(value)}"
                    )
                elif isinstance(value, TensorSpec):
                    nbytes = value.nbytes
                    shape, dtype = value.shape, value.dtype
                    objects_map[key] = SharedMemoryNumpyObject(
                        self._manager.SharedMemory(value.nbytes), value.shape, str(dtype)
                    )
                    logger.debug(
                        f"Created shm segment: key={f'{id}/{key}'}, size={nbytes / 1024 / 1024:.2f} MB, shape={shape}, dtype={dtype}, len=1"
                    )
                else:
                    raise ValueError(f"Unsupported type: {type(value)}")

        self._objects_map.update({f"{namespace}/{k}": v for k, v in objects_map.items()})
        return objects_map

    def cleanup(self, namespace: str = ""):
        """Cleanup the shared memory segments."""
        for key in list(self._objects_map.keys()):
            if namespace == "" or key.startswith(namespace):
                del self._objects_map[key]

    @staticmethod
    def copy_to(shm_map: Dict[str, SharedMemory], data: Dict[str, Any]) -> Dict[str, Any]:
        """Copy the data dict values to the shared memory segments for transport.

        Args:
            data (Dict[str, Any]): Inputs to the model ("images", "texts", "prompts" etc) as
                numpy arrays or lists of numpy arrays.
        Returns:
            Dict[str, Any]: Shared memory segments for the data dict values.
        """
        assert len(shm_map) > 0, "Shared memory segments should not be empty."
        assert len(data) > 0, "Data dict should not be empty."

        # Copy the data dict values to the shared memory segments.
        for key, value in data.items():
            if key in shm_map:
                if isinstance(value, list):
                    if len(value) != len(shm_map[key]):
                        raise ValueError(
                            f"Shared memory already initialized with length={len(shm_map[key])}, provided input with length={len(value)}."
                        )
                    for item, shm in zip(value, shm_map[key]):
                        shm.copy(item)
                elif isinstance(value, np.ndarray):
                    shm_map[key].copy(value)
                else:
                    raise ValueError(f"Unsupported type: {type(value)}")
                # Overwrite the data dict value with the shared memory segments for transport.
                data[key] = shm_map[key]
        logger.debug(f"Successfully copied inputs to shared memory: {shm_map.keys()}")
        return data
