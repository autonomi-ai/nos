from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Tuple


class MethodType(Enum):
    TXT2IMG = "txt2img"
    TXT2VEC = "txt2vec"
    IMG2VEC = "img2vec"
    IMG2BBOX = "img2bbox"


@dataclass(frozen=True)
class ModelSpec:
    """Model specification for the registry.

    The ModelSpec defines all the relevant information for
    the compilation, deployment, and serving of a model.
    """

    name: str
    """Model name."""
    method: MethodType
    """Model method type (e.g. txt2img, txt2vec, img2vec)."""
    cls: Any
    """Model class instance."""
    args: Tuple[Any, ...]
    """Arguments to initialize the model instance."""
    kwargs: Dict[str, Any]
    """Keyword arguments to initialize the model instance."""

    def create(self, *args, **kwargs) -> Any:
        """Create a model instance."""
        return self.cls(*self.args, **self.kwargs)
