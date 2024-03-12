from typing import Any

import cloudpickle


def dumps(obj: Any, **kwargs):
    return cloudpickle.dumps(obj, **kwargs)


def loads(obj: Any, *args, **kwargs):
    return cloudpickle.loads(obj, **kwargs)
