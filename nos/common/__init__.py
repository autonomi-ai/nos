from typing import Any

import cloudpickle

from .spec import FunctionSignature, ModelSpec  # noqa: F401
from .tasks import TaskType  # noqa: F401
from .types import EmbeddingSpec, ImageSpec, TensorSpec  # noqa: F401


def dumps(obj: Any):
    return cloudpickle.dumps(obj, protocol=4)
