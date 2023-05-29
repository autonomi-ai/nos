from dataclasses import field
from typing import Any, Callable, Dict, Optional, Tuple, Type, Union

from pydantic import validator
from pydantic.dataclasses import dataclass

from nos.common.cloudpickle import dumps, loads
from nos.common.tasks import TaskType
from nos.common.types import Batch, EmbeddingSpec, ImageSpec, TensorSpec  # noqa: F401


@dataclass
class FunctionSignature:
    """Function signature."""

    # TOFIX (spillai): Remove Any type, and explicitly define input/output types.
    inputs: Dict[str, Union[Type[int], Type[str], Type[float], Any]]
    """Mapping of input names to dtypes."""
    outputs: Dict[str, Union[Type[int], Type[str], Type[float], Any]]
    """Mapping of output names to dtypes."""

    """The remaining private fields are used to instantiate a model and execute it."""
    func_or_cls: Optional[Callable] = None
    """Class instance."""
    init_args: Tuple[Any, ...] = field(default_factory=tuple)
    """Arguments to initialize the model instance."""
    init_kwargs: Dict[str, Any] = field(default_factory=dict)
    """Keyword arguments to initialize the model instance."""
    method_name: str = None
    """Class method name. (e.g. forward, __call__ etc)"""

    def validate_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the input dict against the defined signature and decode it."""
        if len(set(self.inputs.keys()).symmetric_difference(set(inputs.keys()))) > 0:
            raise ValueError(f"Invalid inputs, provided={set(inputs.keys())}, expected={set(self.inputs.keys())}.")
        # TODO (spillai): Validate input types and shapes.
        return inputs

    def encode_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Encode inputs based on defined signature."""
        inputs = self.validate_inputs(inputs)
        return {k: dumps(v) for k, v in inputs.items()}

    def decode_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Decode inputs based on defined signature."""
        inputs = self.validate_inputs(inputs)
        return {k: loads(v) for k, v in inputs.items()}


@dataclass
class ModelSpec:
    """Model specification for the registry.

    The ModelSpec defines all the relevant information for
    the compilation, deployment, and serving of a model.
    """

    name: str
    """Model name."""
    task: TaskType
    """Task type (e.g. image_embedding, image_generation, object_detection_2d, etc)."""
    signature: FunctionSignature = None
    """Model function signature."""

    @staticmethod
    def get_id(model_name: str, task: TaskType = None) -> str:
        if task is None:
            return model_name
        return f"{task.value}/{model_name}"

    @property
    def id(self) -> str:
        return self.get_id(self.name, self.task)

    @validator("signature")
    def _validate_signature(cls, sig: FunctionSignature, **kwargs: Dict[str, Any]) -> FunctionSignature:
        """Validate the model signature.

        Checks that the model class `cls` has the function name attribute
        as defined in the signature `function_name`.

        Args:
            sig (ModelSignature): Model signature.
            **kwargs: Keyword arguments.
        Returns:
            FunctionSignature: Function signature.
        """
        if sig and sig.func_or_cls:
            model_cls = sig.func_or_cls
            if sig.method_name and not hasattr(model_cls, sig.method_name):
                raise ValueError(f"Model class {model_cls} does not have function {sig.method_name}.")
        return sig

    def create(self, *args, **kwargs) -> Any:
        """Create a model instance."""
        return self.cls(*self.args, **self.kwargs)
