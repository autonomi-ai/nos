import os
import secrets
import time
from dataclasses import dataclass, field
from multiprocessing import resource_tracker
from multiprocessing.managers import SharedMemoryManager
from multiprocessing.shared_memory import SharedMemory
from typing import Any, Dict, Optional, Tuple

import numpy as np

from nos.common.cloudpickle import dumps, loads
from nos.common.types import TensorSpec
from nos.logging import logger


NOS_SHM_ENABLED = bool(int(os.environ.get("NOS_SHM_ENABLED", "1")))
if not NOS_SHM_ENABLED:
    logger.warning("Shared memory transport is disabled.")


@dataclass
class SharedMemoryNumpyObject:
    """Shared memory object wrapping numpy array.

    Shared memory objects are updated with user-permissions (0666) under
    /dev/shm/nos_psm_<random_hex_string> and are automatically cleaned up
    when the object is garbage collected.
    """

    nbytes: int
    """numpy array nbytes"""
    shape: Tuple[int, ...]
    """numpy array shape"""
    dtype: np.dtype
    """numpy array dtype"""
    mode: str = field(init=False, default="r")
    """Shared memory  mode"""
    _shm: SharedMemory = field(init=False, default=None)
    """Shared memory object"""
    _shm_arr: np.ndarray = field(init=False, default=None)

    def __post_init__(self) -> None:
        # Set user-level permissions on the shared memory object
        self._shm = SharedMemory(name=f"nos_psm_{secrets.token_hex(8)}", create=True, size=self.nbytes)
        # TOFIX (spillai): This is a hack to get around the fact that the shared memory
        # object is created with the default permissions (0600) and the user running
        # the inference service is not the same as the user running the client.
        self._shm._fd = os.dup(self._shm._fd)
        os.chown(self._shm._fd, 1000, 1000)
        os.chmod(self._shm._fd, 0o666)
        self.mode = "w"
        # Create a numpy array view of the shared memory object
        self._shm_arr = np.ndarray(shape=self.shape, dtype=self.dtype, buffer=self._shm.buf)

    def __repr__(self) -> str:
        """Return the shared memory object representation."""
        return f"ShmObject(name={self.name}, shape={self.shape}, dtype={self.dtype})"

    def __getstate__(self) -> Dict[str, Any]:
        """Return the shared memory object state.

        This method is called when the object is pickled (dumps).
        """
        return {"name": self.name, "shape": self.shape, "dtype": str(self.dtype)}

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Set the shared memory object state.
        This method is called when the object is unpickled (loads).

        Args:
            state (Dict[str, Any]): Shared memory object state.
        """
        self._shm = SharedMemory(name=state["name"], create=False)
        self.shape = state["shape"]
        assert isinstance(state["dtype"], str)
        self.dtype = np.dtype(state["dtype"])
        self.mode = "r"
        # Create a numpy array view of the shared memory object
        self._shm_arr = np.ndarray(shape=self.shape, dtype=self.dtype, buffer=self._shm.buf)

    def cleanup(self) -> None:
        """Close and unlink the shared memory object (server-side / writer)."""
        self._shm.close()
        self._shm.unlink()
        self._shm_arr = None

    def close(self) -> None:
        """Close the shared memory object (client-side / reader)."""
        # Note (spillai): We need to explicitly call `unregister()` here
        # to avoid the resource tracker from raising a UserWarning about leaked
        # resources. This is because the shared memory implementation in Python
        # assumes that all clients of a segment are child processes from a single
        # parent, and that they inherit the same resource_tracker.
        self._shm.close()
        self._shm_arr = None
        resource_tracker.unregister(self._shm._name, "shared_memory")

    @property
    def name(self) -> str:
        """Return the shared memory name."""
        return self._shm.name

    def copy_from(self, arr: np.ndarray) -> None:
        """Copy data from the numpy array to shared memory object."""
        assert arr.shape == self.shape, f"Array shape {arr.shape} does not match shared memory shape {self.shape}"
        assert arr.dtype == self.dtype, f"Array dtype {arr.dtype} does not match shared memory dtype {self.dtype}"
        self._shm_arr[:] = arr[:]

    def get(self) -> np.ndarray:
        """Get the numpy array from the shared memory object ."""
        return self._shm_arr.copy()


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
        st = time.perf_counter()
        data = {k: loads(v) for k, v in data.items()}
        if NOS_SHM_ENABLED:
            logger.debug(f"Loaded shm dict [elapsed={(time.perf_counter() - st) * 1e3:.1f}ms]")
            st = time.perf_counter()
            shm_keys = set()
            for k, v in data.items():
                if isinstance(v, SharedMemoryNumpyObject):
                    data[k] = v.get()
                    shm_keys.add(k)
            logger.debug(f"Decoded shm data [keys={shm_keys}, elapsed={(time.perf_counter() - st) * 1e3:.1f}ms]")
        return data

    @staticmethod
    def encode(data: Dict[str, Any]) -> Dict[str, Any]:
        """Encode the data dictionary with shared-memory references."""
        return dumps(data)


@dataclass
class SharedMemoryTransportManager:
    """Shared memory transport manager."""

    _shm_manager: SharedMemoryManager = field(init=False, default=None)
    """Shared memory manager."""
    _objects_map: Dict[str, Any] = field(default_factory=dict)
    """Shared memory objects map."""
    _shm_counter: int = field(init=False, default=0)
    """Shared memory counter."""
    _max_rate: float = field(init=False, default=10)
    """Maximum shared memory allocation rate."""
    _last_polled: float = field(init=False, default_factory=lambda: time.time())
    """Last time the shared memory allocation rate was polled."""

    def __post_init__(self) -> None:
        """Initialize the shared memory transport manager."""
        logger.debug("Initializing shared memory transport manager")

    def __del__(self) -> None:
        """Cleanup the shared memory transport manager."""
        logger.debug("Cleaning up shared memory transport manager")
        self.cleanup()

    def create(self, data: Dict[str, Any], namespace: Optional[str] = None) -> Dict[str, Any]:
        """Create a shared memory segment for the data dictionary.

        Note: The keys for shared memory segments are prefixed with the
        namespace `<client_id>/<object_id>/<key>`, while the `objects_map`
        returned does not have the namespace prefixed (i.e. <key>)

        Args:
            data (Dict[str, Any]): Inputs to the model ("images", "texts", "prompts" etc) as
                a dictionary of numpy arrays.
            namespace (str, optional): Unique namespace for the shared memory segment. Defaults to "".
        Returns:
            Dict[str, Any]: Shared memory segment for the data dictionary.
        """
        namespace = namespace or ""

        # Update the number of shm allocations, and rate-limit
        self._shm_counter += 1
        if self._shm_counter % 10 == 0:
            rate = self._shm_counter / (time.time() - self._last_polled)
            if rate > self._max_rate:
                logger.warning(
                    f"Shared memory allocation rate is high, check for variable input shapes with every request "
                    f"[allocation calls={self._shm_counter}, rate={rate:.1f} calls/s]"
                )
            self._last_polled = time.time()
            self._shm_counter = 0

        # Create shared memory segments for numpy arrays (or lists of numpy arrays)
        objects_map: Dict[str, Any] = {}
        for key, value in data.items():
            full_key = f"{namespace}/{key}"
            assert full_key not in self._objects_map, f"Shared memory segment {full_key} already exists."

            if isinstance(value, TensorSpec):
                objects_map[key] = SharedMemoryNumpyObject(
                    value.nbytes,
                    value.shape,
                    np.dtype(value.dtype),
                )
                logger.debug(
                    f"Created shm segment [key={full_key}, size={value.nbytes / 1024 / 1024:.2f} MB, shape={value.shape}, dtype={value.dtype}, len=1]"
                )
            else:
                logger.debug("Ignoring non-tensor input")

        self._objects_map.update({f"{namespace}/{key}": value for key, value in objects_map.items()})
        return objects_map

    def cleanup(self, namespace: Optional[str] = None) -> None:
        """Cleanup the shared memory segments."""
        for key in list(self._objects_map.keys()):
            logger.debug(f"Cleaning up shm segment [key={key}]")
            if namespace is None or key.startswith(namespace):
                # Note (spillai): We need to explicitly call `cleanup()` here
                # as the shared memory segments in order to clean up the shared
                # memory segments immediately after being unregistered.
                self._objects_map[key].cleanup()
                del self._objects_map[key]
                logger.debug(f"Removed shm segment [key={key}]")

    @staticmethod
    def copy(shm_map: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Copy the data dict values to the shared memory segments for transport.

        Args:
            shm_map (Dict[str, SharedMemoryNumpyObject]): Shared memory segments for the data
                dict values (destination).
            data (Dict[str, Any]): Inputs to the model ("images", "texts", "prompts" etc) as
                numpy arrays or lists of numpy arrays (src).
        Returns:
            Dict[str, Any]: Shared memory segments for the data dict values.
        """
        assert len(shm_map) > 0, "Shared memory segments should not be empty."
        assert len(data) > 0, "Data dict should not be empty."

        # Copy the data dict values to the shared memory segments i.e. memcpy(dest, src).
        st = time.perf_counter()
        for key in shm_map.keys():
            assert key in data, f"Key {key} not found in data dict."
            if isinstance(data[key], list):
                assert len(data[key]) == len(
                    shm_map[key]
                ), f"Shared memory already initialized with length={len(shm_map[key])}, provided input with length={len(data[key])}."
                # Move data from the data dict value to the shared memory segment.
                for item, shm in zip(data[key], shm_map[key]):
                    shm.copy_from(item)
            elif isinstance(data[key], np.ndarray):
                # Move data from the data dict value to the shared memory segment.
                shm_map[key].copy_from(data[key])
            else:
                raise ValueError(f"Unsupported type [type={type(data[key])}]")

            # Overwrite the data dict value with the shared memory segments for transport.
            data[key] = shm_map[key]
        logger.debug(f"Copied inputs to shm [keys={shm_map.keys()}, elapsed={(time.perf_counter() - st) * 1e3:.1f}ms]")
        return data
