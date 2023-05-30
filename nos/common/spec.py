import copy
from dataclasses import field
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, get_args, get_origin

from pydantic import validator
from pydantic.dataclasses import dataclass

from nos.common.cloudpickle import dumps, loads
from nos.common.tasks import TaskType
from nos.common.types import Batch, EmbeddingSpec, ImageSpec, ImageT, TensorSpec, TensorT  # noqa: F401
from nos.protoc import import_module


nos_service_pb2 = import_module("nos_service_pb2")


class ObjectTypeInfo:
    """Function signature information.

    Parameters:
        annotated_type (Any): Annotated type.

    Attributes:
        _is_batched (bool): Batched flag.
        _batch_size (int): Batch size.
        _base_type (Any): Base type (Image.Image, np.ndarray etc).
        _base_spec (Any): Base type specification (None, ImageSpec, TensorSpec etc).
    """

    def __init__(self, annotated_type: Any):
        """Initialize the function signature information."""
        self._annotated_type = annotated_type
        try:
            (annotated_cls,) = annotated_type.__args__
        except AttributeError:
            annotated_cls = annotated_type

        # Parse Batch annotation
        self._is_batched, self._batch_size = False, None
        if annotated_cls == Batch:
            annotated_type, batch_size = annotated_type.__metadata__
            self._is_batched, self._batch_size = True, batch_size
            try:
                (annotated_cls,) = annotated_type.__args__
            except AttributeError:
                annotated_cls = annotated_type

        # Parse Tensor/type annotation
        if annotated_cls in (TensorT, ImageT):
            object_type, object_spec = annotated_type.__metadata__
        else:
            try:
                (object_type,) = annotated_type.__metadata__
            except AttributeError:
                object_type = annotated_cls
            object_spec = None

        self._base_type = object_type
        self._base_spec = object_spec

    def __repr__(self) -> str:
        return (
            f"""{self.__class__.__name__}(is_batched={self._is_batched}, batch_size={self._batch_size}, """
            f"""base_type={self._base_type}, base_spec={self._base_spec})"""
        )

    def is_batched(self) -> bool:
        """Return the `is_batched` flag.

        Returns:
            bool: Flag to indicate if batching is enabled.
                If true, `batch_size=None` implies dynamic batch size, otherwise `batch_size=<int>`.
        """
        return self._is_batched

    def batch_size(self) -> int:
        """Return the batch size.

        Returns:
            int: Batch size. If `None` and `is_batched` is `true`, then batch size is considered dynamic.
        """
        return self._batch_size

    def base_type(self) -> Any:
        """Return the base type.

        Returns:
            Any: Base type. Base type here can be simple types (e.g. `str`, `int`, ...) or
                complex types with library dependencies (e.g. `np.ndarray`, `PIL.Image.Image` etc).
        """
        return self._base_type

    def base_spec(self) -> Optional[Union[TensorSpec, ImageSpec, EmbeddingSpec]]:
        """Return the base spec.

        Returns:
            Optional[Union[TensorSpec, ImageSpec, EmbeddingSpec]]: Base spec.
        """
        return self._base_spec


def parse_annotated_type(annotated_type: Any) -> Union[ObjectTypeInfo, List[ObjectTypeInfo]]:
    """Parse annotated type."""
    # Union of annotated types are converted into set of annotated types.
    if get_origin(annotated_type) == Union:
        return [parse_annotated_type(t) for t in get_args(annotated_type)]
    return ObjectTypeInfo(annotated_type)


@dataclass
class FunctionSignature:
    """Function signature that fully describes the remote-model to be executed
    including `inputs`, `outputs`, `func_or_cls` to be executed,
    initialization `args`/`kwargs`."""

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

    def __repr__(self) -> str:
        """Return the function signature representation."""
        return f"FunctionSignature(inputs={self.inputs}, outputs={self.outputs}, func_or_cls={self.func_or_cls}, init_args={self.init_args}, init_kwargs={self.init_kwargs}, method_name={self.method_name})"

    def _validate_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the input dict against the defined signature and decode it."""
        if len(set(self.inputs.keys()).symmetric_difference(set(inputs.keys()))) > 0:
            raise ValueError(f"Invalid inputs, provided={set(inputs.keys())}, expected={set(self.inputs.keys())}.")
        # TODO (spillai): Validate input types and shapes.
        return inputs

    def _encode_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Encode inputs based on defined signature."""
        inputs = self._validate_inputs(inputs)
        return {k: dumps(v) for k, v in inputs.items()}

    def _decode_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Decode inputs based on defined signature."""
        inputs = self._validate_inputs(inputs)
        return {k: loads(v) for k, v in inputs.items()}

    def get_inputs_spec(self) -> Dict[str, Union[ObjectTypeInfo, List[ObjectTypeInfo]]]:
        """Return the full input function signature specification.

        For example, for CLIP's text embedding model, the inputs/output spec is:
            ```
            inputs  = {'texts': ObjectTypeInfo(is_batched=True, batch_size=None, base_type=<class 'str'>, base_spec=None)}
            outputs = {'embedding': ObjectTypeInfo(is_batched=True, batch_size=None, base_type=<class 'numpy.ndarray'>, base_spec=EmbeddingSpec(shape=(512,), dtype='float32'))}
            ```
        Returns:
            Dict[str, Union[ObjectTypeInfo, List[ObjectTypeInfo]]]: Inputs spec.
        """
        return {k: parse_annotated_type(v) for k, v in self.inputs.items()}

    def get_outputs_spec(self) -> Dict[str, Union[ObjectTypeInfo, List[ObjectTypeInfo]]]:
        """Return the full output function signature specification.

        Returns:
            Dict[str, Union[ObjectTypeInfo, List[ObjectTypeInfo]]]: Outputs spec.
        """
        return {k: parse_annotated_type(v) for k, v in self.outputs.items()}


@dataclass
class ModelSpec:
    """Model specification for the registry.

    The ModelSpec defines all the relevant information for
    the compilation, deployment, and serving of a model.
    """

    name: str
    """Model identifier."""
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
        return self.cls(*args, **kwargs)

    def _to_proto(self, public: bool = False) -> nos_service_pb2.ModelInfoResponse:
        """Convert the model spec to proto."""
        if public:
            spec = copy.deepcopy(self)
            spec.func_or_cls = None
            spec.init_args = ()
            spec.init_kwargs = {}
            spec.method_name = None
        else:
            spec = self
        return nos_service_pb2.ModelInfoResponse(
            response_bytes=dumps(spec),
        )
