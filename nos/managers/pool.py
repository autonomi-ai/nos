"""Actor pool based on ray.util.ActorPool with task throttling."""
from typing import TYPE_CHECKING, TypeVar

import ray


if TYPE_CHECKING:
    import ray.actor

V = TypeVar("V")


class ActorPool:
    """Actor pool with task throttling."""

    def __init__(self, actors: list):
        self._idle_actors = list(actors)

        # future to result
        self._future_to_result = {}
        self._future_to_index = {}

        # get actor from future
        self._future_to_actor = {}

        # get future from index
        self._index_to_future = {}

        # next task to do
        self._next_task_index = 0

        # next task to return
        self._next_return_index = 0

    def submit(self, fn, value):
        """Schedule a single task to run in the pool."""
        # If no idle actors, wait for one to become idle
        # This throttles the number of tasks submitted to the pool.
        if not len(self._idle_actors):
            future = self.get_next_unordered()
            self._future_to_result[future] = ray.get(future)

        assert len(self._idle_actors), "No idle actors available"
        actor = self._idle_actors.pop()
        future = fn(actor, value)
        self._future_to_actor[future] = (self._next_task_index, actor)
        self._index_to_future[self._next_task_index] = future
        self._future_to_index[future] = self._next_task_index
        self._next_task_index += 1
        return future

    def has_next(self):
        """Returns whether there are any pending results to return."""
        return bool(self._future_to_actor)

    def get(self, future):
        """Returns the result of a pending task."""
        if future in self._future_to_result:
            return self._future_to_result.pop(future)
        result = ray.get(future)
        index = self._future_to_index.pop(future)
        del self._index_to_future[index]
        i, a = self._future_to_actor.pop(future)
        self._return_actor(a)
        return result

    async def async_get(self, future):
        """Returns the result of a pending task asynchronously."""
        if future in self._future_to_result:
            return self._future_to_result.pop(future)
        result = await future
        index = self._future_to_index.pop(future)
        del self._index_to_future[index]
        i, a = self._future_to_actor.pop(future)
        self._return_actor(a)
        return result

    def get_next(self, timeout=None, ignore_if_timedout=False):
        """Returns the next pending result in order."""
        if not self.has_next():
            raise StopIteration("No more results to get")
        if self._next_return_index >= self._next_task_index:
            raise ValueError("It is not allowed to call get_next() after get_next_unordered().")
        future = self._index_to_future[self._next_return_index]
        timeout_msg = "Timed out waiting for result"
        raise_timeout_after_ignore = False
        if timeout is not None:
            res, _ = ray.wait([future], timeout=timeout)
            if not res:
                if not ignore_if_timedout:
                    raise TimeoutError(timeout_msg)
                else:
                    raise_timeout_after_ignore = True
        del self._index_to_future[self._next_return_index]
        self._next_return_index += 1

        future_key = tuple(future) if isinstance(future, list) else future
        i, a = self._future_to_actor.pop(future_key)

        self._return_actor(a)
        if raise_timeout_after_ignore:
            raise TimeoutError(timeout_msg + ". The task {} has been ignored.".format(future))
        return future

    def get_next_unordered(self, timeout=None, ignore_if_timedout=False):
        """Returns any of the next pending results."""
        if not self.has_next():
            raise StopIteration("No more results to get")
        res, _ = ray.wait(list(self._future_to_actor), num_returns=1, timeout=timeout)
        timeout_msg = "Timed out waiting for result"
        raise_timeout_after_ignore = False
        if res:
            [future] = res
        else:
            if not ignore_if_timedout:
                raise TimeoutError(timeout_msg)
            else:
                raise_timeout_after_ignore = True
        i, a = self._future_to_actor.pop(future)
        self._return_actor(a)
        del self._index_to_future[i]
        self._next_return_index = max(self._next_return_index, i + 1)
        if raise_timeout_after_ignore:
            raise TimeoutError(timeout_msg + ". The task {} has been ignored.".format(future))
        return future

    def _return_actor(self, actor):
        self._idle_actors.append(actor)

    def has_free(self):
        return len(self._idle_actors) > 0
