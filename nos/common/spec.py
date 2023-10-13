import inspect
import re
from dataclasses import asdict, field
from functools import cached_property
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, get_args, get_origin

from pydantic import validator
from pydantic.dataclasses import dataclass

from nos.common.cloudpickle import dumps, loads
from nos.common.exceptions import NosInputValidationException
from nos.common.tasks import TaskType
from nos.common.types import Batch, EmbeddingSpec, ImageSpec, ImageT, TensorSpec, TensorT  # noqa: F401
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

    func_or_cls: Optional[Callable]
    """Class instance."""
    inputs: Dict[str, FunctionSignatureType]
    """Mapping of input names to dtypes."""
    outputs: Dict[str, FunctionSignatureType]
    """Mapping of output names to dtypes."""

    """The remaining private fields are used to instantiate a model and execute it."""
    init_args: Tuple[Any, ...] = field(default_factory=tuple)
    """Arguments to initialize the model instance."""
    init_kwargs: Dict[str, Any] = field(default_factory=dict)
    """Keyword arguments to initialize the model instance."""
    method: str = None
    """Class method name. (e.g. forward, __call__ etc)"""

    def __repr__(self) -> str:
        """Return the function signature representation."""
        return f"FunctionSignature({asdict(self)})"

    @staticmethod
    def validate(inputs: Dict[str, Any], sig: Dict[str, FunctionSignatureType]) -> Dict[str, Any]:
        """Validate the input dict against the defined signature (input or output)."""
        # TOFIX (spillai): This needs to be able to validate using args/kwargs instead
        if not set(inputs.keys()).issubset(set(sig.keys())):  # noqa: W503
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


class ModelSpecMetadataRegistry:
    """Model specification registry."""

    _instance: Optional["ModelSpecMetadataRegistry"] = None
    """Singleton instance."""

    _registry: Dict[str, "ModelSpecMetadata"] = {}
    """Model specification metadata registry."""

    @classmethod
    def get(cls) -> "ModelSpecMetadataRegistry":
        """Get the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __contains__(self, model_id: Any) -> bool:
        """Check if the model spec metadata is available."""
        return model_id in self._registry

    def __getitem__(self, model_id: Any) -> "ModelSpecMetadata":
        """Load the model spec metadata."""
        try:
            return self._registry[model_id]
        except KeyError:
            raise KeyError(f"Unavailable model (id={model_id}).")

    def __setitem__(self, model_id: Any, metadata: "ModelSpecMetadata"):
        """Add the model spec metadata."""
        self._registry[model_id] = metadata

    def load(self, model_id: Any) -> "ModelSpec":
        """Load the model spec metadata (identical to __getitem__)."""
        return self[model_id]


@dataclass
class ModelSpecMetadata:
    """Model specification metadata.

    The metadata contains the model profiles, metrics, etc.
    """

    id: str
    """Model identifier."""
    method: str
    """Model method name."""
    task: TaskType = None
    """Task type (e.g. image_embedding, image_generation, object_detection_2d, etc)."""
    resources: Dict[str, ModelResources] = field(default_factory=dict)
    """Model resource limits (device/host memory, etc)."""
    """Key is the runtime type (cpu, gpu, trt-runtime, etc)."""

    def __repr__(self) -> str:
        return (
            f"""ModelSpecMetadata(id={self.id}, task={self.task}, method={self.method}, resources={self.resources})"""
        )


@dataclass
class ModelSpec:
    """Model specification for the registry.

    ModelSpec captures all the relevant information for
    the instantiation, runtime and execution of a model.
    """

    id: str
    """Model identifier."""
    signature: Dict[str, FunctionSignature] = field(default_factory=dict)
    """Model function signatures to export (method -> FunctionSignature)."""
    runtime_env: RuntimeEnv = None
    """Runtime environment with custom packages."""

    def __post_init__(self):
        # Model metadata (method -> ModelSpecMetadata)
        self._metadata: Dict[str, ModelSpecMetadata] = None

    @validator("id", pre=True)
    def _validate_id(cls, id: str) -> str:
        """Validate the model identifier."""
        regex = re.compile(r"^[a-zA-Z0-9\/._-]+$")  # allow alphanumerics, `/`, `.`, `_`, and `-`
        if not regex.match(id):
            raise ValueError(
                f"Invalid model id, id={id} can only contain alphanumerics characters, `/`, `.`, `_`, and `-`"
            )
        return id

    def __repr__(self):
        return f"""ModelSpec(id={self.id}, methods=[{', '.join(list(self.signature))}]"""

    @validator("signature", pre=True)
    def _validate_signature(
        cls, sigs: List[FunctionSignature], **kwargs: Dict[str, Any]
    ) -> Dict[str, FunctionSignature]:
        """Validate the model signature / signatures.

        Checks that the model class `cls` has the function name attribute
        as defined in the signature `function_name`.

        Args:
            sigs (Dict[str, FunctionSignature]): Model signature.
            **kwargs: Keyword arguments.
        Returns:
            Dict[str, FunctionSignature]: Model signature.
        """
        if isinstance(sigs, FunctionSignature):
            sigs = [sigs]
        assert isinstance(sigs, (list, tuple)), f"Invalid signature provided, signature={sigs}."

        signatures: Dict[str, FunctionSignature] = {}
        for sig in sigs:
            if sig and sig.func_or_cls:
                model_cls = sig.func_or_cls
                if sig.method and not hasattr(model_cls, sig.method):
                    raise ValueError(f"Model class {model_cls} does not have function {sig.method}.")
                signatures[sig.method] = sig
        return signatures

    @property
    def name(self) -> str:
        """Return the model name (for backwards compatibility)."""
        return self.id

    def task(self, method: str = None) -> TaskType:
        """Return the task type for a given method (or defaults to default method)."""
        if method is None:
            method = self.default_method
        try:
            md = self.metadata(method)
            return md.task
        except Exception:
            logger.warning(f"Model metadata not found, id={self.id}.")
            return None

    def metadata(self, method: str = None) -> ModelSpecMetadata:
        """Return the model spec metadata for a given method (or defaults to default method)."""
        if method is None:
            method = self.default_method
        try:
            return ModelSpecMetadataRegistry.get()[(self.id, method)]
        except KeyError:
            logger.warning(f"Model metadata not found, id={self.id}.")
            return None

    @cached_property
    def default_method(self) -> str:
        """Return the default method name."""
        assert len(self.signature) > 0, f"No default signature found, signature={self.signature}."
        return list(self.signature.keys())[0]

    @cached_property
    def default_signature(self) -> FunctionSignature:
        """Return the default function signature.

        Returns:
            FunctionSignature: Default function signature.
        """
        # Note (spillai): For now, we assume that the first
        # signature is the default signature. In the `.from_cls()`
        # method, we add the __call__ method as the first method
        # for this exact reason.
        return self.signature[self.default_method]

    def __call__(self, *init_args, **init_kwargs) -> Any:
        """Create a model instance.

        This method allows us to create a model instance directly
        from the model spec. Let's consider the example below

            ```
            class CustomModel:
                ...

            CustomModel = ModelSpec.from_cls(CustomModel)
            model = CustomModel(*init_args, **init_kwargs)
            ```

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        Returns:
            Any: Model instance.
        """
        sig: FunctionSignature = self.default_signature
        return sig.func_or_cls(*init_args, **init_kwargs)

    @classmethod
    def from_cls(
        cls, func_or_cls: Callable, method: str = "__call__", runtime_env: RuntimeEnv = None, **kwargs: Any
    ) -> "ModelSpec":
        """Wrap custom models/classes into a nos-compatible model spec.

        Args:
            func_or_cls (Callable): Model function or class. For now, only classes are supported.
            method (str): Method name to be executed.
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
        if not hasattr(func_or_cls, method):
            raise ValueError(f"Invalid method name `{method}` provided.")

        # TODO (spillai): Provide additional RayRuntimeEnvConfig as `config`
        # config = dict(setup_timeout_seconds=10 * 60, eager_install=True)
        if runtime_env:
            logger.debug(f"Using custom runtime_env [env={runtime_env}]")

        # Inspect all the public methods of the class
        # and expose them as model methods
        ignore_methods = ["__init__", method]
        all_methods = [m for m, _ in inspect.getmembers(func_or_cls, predicate=inspect.isfunction)]
        methods = [m for m in all_methods if m not in ignore_methods]

        # Note (spillai): See .default_signature property for why we add
        #  the __call__ method as the first method.
        if method in all_methods:
            methods.insert(0, method)  # first method is the default method
        logger.debug(f"Registering methods [methods={methods}].")

        # Add function signature for each method
        signatures: List[FunctionSignature] = []
        for method in methods:
            # Get the function signature
            sig = inspect.signature(getattr(func_or_cls, method))

            # Get the positional arguments and their types
            # Note: We skip the `self` argument
            call_inputs = {
                k: v.annotation for k, v in sig.parameters.items() if v.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
            }
            call_inputs.pop("self", None)

            # Get the return type
            call_outputs = {"result": sig.return_annotation}

            # Add the function signature
            sig = FunctionSignature(
                func_or_cls,
                method=method,
                inputs=call_inputs,
                outputs=call_outputs,
            )
            signatures.append(sig)
            logger.debug(f"Added function signature [method={method}, signature={sig}].")

        # Build the model spec from the function signature
        spec = cls(
            func_or_cls.__name__,
            signature=signatures,
            runtime_env=runtime_env,
        )
        return spec

    def _to_proto(self) -> nos_service_pb2.GenericResponse:
        """Convert the model spec to proto."""
        return nos_service_pb2.GenericResponse(
            response_bytes=dumps(self),
        )

    @staticmethod
    def _from_proto(minfo: nos_service_pb2.GenericResponse) -> "ModelSpec":
        """Convert the generic response back to the spec."""
        return loads(minfo.response_bytes)
