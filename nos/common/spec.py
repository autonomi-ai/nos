import copy
import inspect
import json
from dataclasses import asdict, field
from functools import cached_property
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, get_args, get_origin

from pydantic import validator
from pydantic.dataclasses import dataclass

from nos.common.cloudpickle import dumps, loads
from nos.common.exceptions import NosInputValidationException
from nos.common.tasks import TaskType
from nos.common.types import Batch, EmbeddingSpec, ImageSpec, ImageT, TensorSpec, TensorT  # noqa: F401
from nos.constants import NOS_MODELS_DIR
from nos.logging import logger
from nos.protoc import import_module


# TOFIX (spillai): Remove Any type, and explicitly define input/output types.
FunctionSignatureType = Union[Type[int], Type[str], Type[float], Any]

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

    inputs: Dict[str, FunctionSignatureType]
    """Mapping of input names to dtypes."""
    outputs: Dict[str, FunctionSignatureType]
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
        inputs_str = "\n".join([f"{k}={v}" for k, v in self.inputs.items()])
        outputs_str = "\n".join([f"{k}={v}" for k, v in self.outputs.items()])
        return (
            f"""FunctionSignature\n"""
            f"""\tfunc_or_cls={self.func_or_cls}\n"""
            f"""\tinit_args={self.init_args}, init_kwargs={self.init_kwargs}\n"""
            f"""\tmethod_name={self.method_name}\n"""
            f"""\tinputs={inputs_str}\n"""
            f"""\toutputs={outputs_str}\n"""
        )

    @staticmethod
    def validate(inputs: Dict[str, Any], sig: Dict[str, FunctionSignatureType]) -> Dict[str, Any]:
        """Validate the input dict against the defined signature (input or output)."""
        if len(set(sig.keys()).symmetric_difference(set(inputs.keys()))) > 0:
            raise NosInputValidationException(
                f"Invalid inputs, provided={set(inputs.keys())}, expected={set(sig.keys())}."
            )
        # TODO (spillai): Validate input types and shapes.
        return inputs

    def _encode_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Encode inputs based on defined signature."""
        inputs = FunctionSignature.validate(inputs, self.inputs)
        return {k: dumps(v) for k, v in inputs.items()}

    def _decode_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Decode inputs based on defined signature."""
        inputs = FunctionSignature.validate(inputs, self.inputs)
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
class RuntimeEnv:
    conda: Dict[str, Any]
    """Conda environment specification."""

    @classmethod
    def from_packages(cls, packages: List[str]) -> Dict[str, Any]:
        return cls(conda={"dependencies": ["pip", {"pip": packages}]})


@dataclass
class ModelResources:
    """Model resources (device/host memory etc)."""

    runtime: str = "cpu"
    """Runtime type (cpu, gpu, trt-runtime, etc).
    See `nos.server._runtime.InferenceServiceRuntime` for the list of supported runtimes.
    """
    device: str = "cpu"
    """Device type (cpu, cuda, mps, neuron, etc)."""
    device_memory: Union[int, str] = field(default=512 * 1024**2)
    """Device memory (defaults to 512 MB)."""
    cpus: float = 0
    """Number of CPUs (defaults to 0 CPUs)."""
    memory: Union[int, str] = field(default=256 * 1024**2)
    """Host memory (defaults to 256 MB)"""

    @validator("runtime")
    def _validate_runtime(cls, runtime: str) -> str:
        """Validate the runtime."""
        from nos.server._runtime import InferenceServiceRuntime

        # Check if runtime is subset of supported runtimes.
        if runtime not in InferenceServiceRuntime.configs.keys():
            raise ValueError(f"Invalid runtime, runtime={runtime}.")
        return runtime

    @validator("device")
    def _validate_device(cls, device: str) -> str:
        """Validate the device."""
        from nos.server._runtime import NOS_SUPPORTED_DEVICES

        if device not in NOS_SUPPORTED_DEVICES:
            err_msg = f"Invalid device provided, device={device}. Use one of {NOS_SUPPORTED_DEVICES}."
            logger.error(err_msg)
            raise ValueError(err_msg)
        return device

    @validator("device_memory")
    def _validate_device_memory(cls, device_memory: Union[int, str]) -> int:
        """Validate the device memory."""
        if isinstance(device_memory, str):
            raise NotImplementedError()

        if device_memory < 256 * 1024**2 or device_memory > 128 * 1024**3:
            err_msg = f"Invalid device memory provided, device_memory={device_memory / 1024**2} MB. Provide a value between 256 MB and 128 GB."
            logger.error(err_msg)
            raise ValueError(err_msg)
        return device_memory

    @validator("cpus")
    def _validate_cpus(cls, cpus: Union[float, str]) -> float:
        """Validate the number of CPUs."""
        if isinstance(cpus, str):
            raise NotImplementedError()

        if cpus < 0.0 or cpus > 128.0:
            err_msg = f"Invalid number of CPUs provided, cpus={cpus}. Provide a value between 0 and 128."
            logger.error(err_msg)
            raise ValueError(err_msg)
        return cpus

    @validator("memory")
    def _validate_memory(cls, memory: Union[int, str]) -> int:
        """Validate the host memory."""
        if isinstance(memory, str):
            raise NotImplementedError()

        if memory < 256 * 1024**2 or memory > 128 * 1024**3:
            err_msg = f"Invalid device memory provided, memory={memory / 1024**2} MB. Provide a value between 256 MB and 128 GB."
            logger.error(err_msg)
            raise ValueError(err_msg)
        return memory


@dataclass
class ModelSpecMetadata:
    """Model specification metadata.

    The metadata contains the model profiles, metrics, etc.
    """

    name: str
    """Model identifier."""
    task: TaskType
    """Task type (e.g. image_embedding, image_generation, object_detection_2d, etc)."""
    resources: Dict[str, ModelResources] = field(default_factory=dict)
    """Model resource limits (device/host memory, etc)."""
    """Key is the runtime type (cpu, gpu, trt-runtime, etc)."""

    def __repr__(self) -> str:
        return f"""ModelSpecMetadata(name={self.name}, task={self.task}, """ f"""resources={self.resources})"""

    def to_json(self, filename: str) -> Dict[str, Any]:
        """Convert the model spec to json."""
        specd = asdict(self)
        with open(filename, "w") as f:
            json.dump(specd, f, indent=4)
        return specd

    @classmethod
    def from_json(cls, filename: str) -> "ModelSpecMetadata":
        """Convert the model spec from json."""
        with open(filename, "r") as f:
            specd = json.load(f)
            return cls(**specd)


def _metadata_path(spec: "ModelSpec") -> str:
    """Return the metadata path for a model."""
    return NOS_MODELS_DIR / f"metadata/{spec.id}/metadata.json"


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
    runtime_env: RuntimeEnv = None
    """Runtime environment with custom packages."""
    metadata_: ModelSpecMetadata = field(init=False, default=None)
    """Model specification metadata. The contents of the metadata (profiles, metrics, etc)
    are specified in a separate file."""

    class Config:
        """Custom configuration to enable private attributes."""

        underscore_attrs_are_private: bool = True

    def __repr__(self):
        return f"""ModelSpec(name={self.name}, task={self.task})""" f"""\n    {self.signature}"""

    @cached_property
    def metadata(self) -> ModelSpecMetadata:
        try:
            path = _metadata_path(self)
            if not path.exists():
                raise FileNotFoundError(f"Model metadata not found. [path={path}]")
            metadata = ModelSpecMetadata.from_json(str(path))
            logger.info(f"Loaded model metadata [name={self.name}, path={path}, metadata={metadata}]")
        except Exception:
            metadata = None
        return metadata

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
            spec.signature.func_or_cls = None
            spec.signature.init_args = ()
            spec.signature.init_kwargs = {}
            spec.signature.method_name = None
        else:
            spec = self
        return nos_service_pb2.ModelInfoResponse(
            response_bytes=dumps(spec),
        )

    @staticmethod
    def _from_proto(minfo: nos_service_pb2.ModelInfoResponse) -> "ModelSpec":
        """Convert the model info response back to the spec."""
        return loads(minfo.response_bytes)

    @classmethod
    def from_cls(
        cls, func_or_cls: Callable, method_name: str = "__call__", runtime_env: RuntimeEnv = None, **kwargs: Any
    ) -> "ModelSpec":
        """Wrap custom models/classes into a nos-compatible model spec.

        Args:
            func_or_cls (Callable): Model function or class. For now, only classes are supported.
            method_name (str): Method name to be executed.
            runtime_env (RuntimeEnv): Runtime environment specification.
            **kwargs: Additional keyword arguments.
                These include `init_args` and `init_kwargs` to initialize the model instance.

        Returns:
            ModelSpec: The resulting model specification that fully describes the model execution.
        """
        # Check if the cls is not a function
        if not callable(func_or_cls) or not inspect.isclass(func_or_cls):
            raise ValueError(f"Invalid class `{func_or_cls}` provided, needs to be a class object.")

        # Check if the cls has the method_name
        if not hasattr(func_or_cls, method_name):
            raise ValueError(f"Invalid method name `{method_name}` provided.")

        # Get the function signature
        sig = inspect.signature(getattr(func_or_cls, method_name))

        # Get the positional arguments and their types
        # Note: We skip the `self` argument
        call_inputs = {
            k: v.annotation for k, v in sig.parameters.items() if v.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
        }
        call_inputs.pop("self", None)

        # Get the return type
        call_outputs = {"result": sig.return_annotation}

        # Build the model spec from the function signature
        # TODO (spillai): Provide additional RayRuntimeEnvConfig as `config`
        # config = dict(setup_timeout_seconds=10 * 60, eager_install=True)
        if runtime_env:
            logger.debug(f"Using custom runtime_env [env={runtime_env}]")
        spec = cls(
            name=func_or_cls.__name__,
            task=TaskType.CUSTOM,
            signature=FunctionSignature(
                func_or_cls=func_or_cls,
                init_args=kwargs.pop("init_args", ()),
                init_kwargs=kwargs.pop("init_kwargs", {}),
                method_name=method_name,
                inputs=call_inputs,
                outputs=call_outputs,
            ),
            runtime_env=runtime_env,
        )
        return spec
