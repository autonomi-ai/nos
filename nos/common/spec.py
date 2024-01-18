import copy
import inspect
import math
import re
from dataclasses import asdict, field
from functools import cached_property
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union, get_args, get_origin

import humanize
from pydantic import validator
from pydantic.dataclasses import dataclass

from nos.common.cloudpickle import dumps, loads
from nos.common.exceptions import InputValidationException
from nos.common.helpers import memory_bytes
from nos.common.runtime import RuntimeEnv
from nos.common.tasks import TaskType
from nos.common.types import Batch, EmbeddingSpec, ImageSpec, ImageT, TensorSpec, TensorT  # noqa: F401
from nos.constants import NOS_PROFILE_CATALOG_PATH
from nos.logging import logger
from nos.protoc import import_module


logger.disable(__name__)
nos_service_pb2 = import_module("nos_service_pb2")


class ObjectTypeInfo:
    """Function signature information.

    Parameters:
        annotation (Any): Annotation for an input/output.
        parameter (inspect.Parameter): Parameter information (optional).

    Attributes:
        _is_batched (bool): Batched flag.
        _batch_size (int): Batch size.
        _base_type (Any): Base type (Image.Image, np.ndarray etc).
        _base_spec (Any): Base type specification (None, ImageSpec, TensorSpec etc).
    """

    def __init__(self, annotation: Any, parameter: inspect.Parameter = None):
        """Initialize the function signature information."""
        self.annotation = annotation
        self.parameter = parameter
        try:
            (annotated_cls,) = annotation.__args__
        except AttributeError:
            annotated_cls = annotation

        # Parse Batch annotation
        self._is_batched, self._batch_size = False, None
        if annotated_cls == Batch:
            annotation, batch_size = annotation.__metadata__
            self._is_batched, self._batch_size = True, batch_size
            try:
                (annotated_cls,) = annotation.__args__
            except AttributeError:
                annotated_cls = annotation

        # Parse Tensor/type annotation
        if annotated_cls in (TensorT, ImageT):
            object_type, object_spec = annotation.__metadata__
        else:
            try:
                (object_type,) = annotation.__metadata__
            except AttributeError:
                object_type = annotated_cls
            object_spec = None

        # Parse the base type and spec
        self._base_type = object_type
        self._base_spec = object_spec

    def __repr__(self) -> str:
        """Return the function signature information representation."""
        repr = (
            f"""{self.__class__.__name__}(is_batched={self._is_batched}, batch_size={self._batch_size}, """
            f"""base_type={self._base_type}, base_spec={self._base_spec})"""
        )
        if self.parameter:
            p_repr = f"pname={self.parameter}, ptype={self.parameter.annotation}, pdefault={self.parameter.default}"
            repr = f"{repr}, {p_repr}"
        return repr

    def parameter_name(self) -> str:
        """Return the parameter name."""
        return self.parameter.name

    def parameter_annotation(self) -> Any:
        """Return the parameter annotation."""
        return self.parameter.annotation

    def parameter_default(self) -> Any:
        """Return the parameter default."""
        return self.parameter.default

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


def AnnotatedParameter(
    annotation: Any, parameter: inspect.Parameter = None
) -> Union[ObjectTypeInfo, List[ObjectTypeInfo]]:
    """Annotate the parameter for inferring additional metdata."""
    # Union of annotated types are converted into set of annotated types.
    if get_origin(annotation) == Union:
        return [AnnotatedParameter(ann, parameter) for ann in get_args(annotation)]
    return ObjectTypeInfo(annotation, parameter)


@dataclass
class FunctionSignature:
    """Function signature that fully describes the remote-model to be executed
    including `inputs`, `outputs`, `func_or_cls` to be executed,
    initialization `args`/`kwargs`."""

    func_or_cls: Callable
    """Class instance."""
    method: str
    """Class method name. (e.g. forward, __call__ etc)"""

    init_args: Tuple[Any, ...] = field(default_factory=tuple)
    """Arguments to initialize the model instance."""
    init_kwargs: Dict[str, Any] = field(default_factory=dict)
    """Keyword arguments to initialize the model instance."""

    parameters: Dict[str, Any] = field(init=False)
    """Input function signature (as returned by inspect.signature)."""
    return_annotation: Any = field(init=False)
    """Output / return function signature (as returned by inspect.signature)."""

    input_annotations: Dict[str, Any] = field(default_factory=dict)
    """Mapping of input keyword arguments to dtypes."""
    output_annotations: Union[Any, Dict[str, Any], None] = field(default=None)
    """Mapping of output names to dtypes."""

    def __post_init__(self):
        if not callable(self.func_or_cls):
            raise ValueError(f"Invalid function/class provided, func_or_cls={self.func_or_cls}.")

        if not self.method or not hasattr(self.func_or_cls, self.method):
            raise ValueError(f"Invalid method name provided, method={self.method}.")

        # Get the function signature
        sig: Dict[str, inspect.Parameter] = inspect.signature(getattr(self.func_or_cls, self.method))

        # Get the input/output annotations
        self.parameters = sig.parameters
        self.return_annotation = sig.return_annotation
        logger.debug(f"Function signature [method={self.method}, sig={sig}].")

    def __repr__(self) -> str:
        """Return the function signature representation."""
        return f"FunctionSignature({asdict(self)})"

    @staticmethod
    def validate(inputs: Dict[str, Any], sig: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the input dict against the defined signature (input or output)."""
        # TOFIX (spillai): This needs to be able to validate using args/kwargs instead
        if not set(inputs.keys()).issubset(set(sig.keys())):  # noqa: W503
            raise InputValidationException(
                f"Invalid inputs, provided={set(inputs.keys())}, expected={set(sig.keys())}."
            )
        # TODO (spillai): Validate input types and shapes.
        return inputs

    @validator("init_args", pre=True)
    def _validate_init_args(cls, init_args: Union[Tuple[Any, ...], Any]) -> Tuple[Any, ...]:
        """Validate the initialization arguments."""
        # TODO (spillai): Check the function signature of the func_or_cls class and validate
        # the init_args against the signature.
        return init_args

    @validator("init_kwargs", pre=True)
    def _validate_init_kwargs(cls, init_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the initialization keyword arguments."""
        # TODO (spillai): Check the function signature of the func_or_cls class and validate
        # the init_kwargs against the signature.
        return init_kwargs

    def _encode_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Encode inputs based on defined signature."""
        inputs = FunctionSignature.validate(inputs, self.parameters)
        return {k: dumps(v) for k, v in inputs.items()}

    def _decode_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Decode inputs based on defined signature."""
        inputs = FunctionSignature.validate(inputs, self.parameters)
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
        parameters = self.parameters.copy()
        parameters.pop("self", None)
        return {k: AnnotatedParameter(self.input_annotations.get(k, p.annotation), p) for k, p in parameters.items()}

    def get_outputs_spec(self) -> Dict[str, Union[ObjectTypeInfo, Dict[str, ObjectTypeInfo]]]:
        """Return the full output function signature specification.

        Returns:
            Dict[str, Union[ObjectTypeInfo, Dict[str, ObjectTypeInfo]]]: Outputs spec.
        """
        if self.output_annotations is None:
            return AnnotatedParameter(self.return_annotation)
        elif isinstance(self.output_annotations, dict):
            return {k: AnnotatedParameter(ann) for k, ann in self.output_annotations.items()}
        else:
            return AnnotatedParameter(self.output_annotations)


@dataclass
class ModelResources:
    """Model resources (device/host memory etc)."""

    runtime: str = field(default="auto")
    """Runtime type (cpu, gpu, trt, etc).
    See `nos.server._runtime.InferenceServiceRuntime` for the list of supported runtimes.
    """
    cpus: float = 0
    """Number of CPUs (defaults to 0 CPUs)."""
    memory: Union[None, int, str] = field(default=0)
    """Host / CPU memory"""
    device: str = field(default="auto")
    """Device identifier (nvidia-2080, nvidia-4090, apple-m2, etc)."""
    device_memory: Union[int, str] = field(default="auto")
    """Device / GPU memory."""

    def __repr__(self) -> str:
        memory = humanize.naturalsize(self.memory, binary=True) if self.memory else None
        device_memory = self.device_memory
        if isinstance(device_memory, int):
            device_memory = humanize.naturalsize(self.device_memory, binary=True) if self.device_memory else None
        assert device_memory is None or isinstance(device_memory, str)
        return (
            f"""ModelResources(runtime={self.runtime}, device={self.device}, cpus={self.cpus}, """
            f"""memory={memory}, device_memory={device_memory})"""
        )

    @validator("runtime")
    def _validate_runtime(cls, runtime: str) -> str:
        """Validate the runtime."""
        from nos.server._runtime import InferenceServiceRuntime

        # Check if runtime is subset of supported runtimes.
        if runtime not in list(InferenceServiceRuntime.configs.keys()) + ["auto"]:
            raise ValueError(f"Invalid runtime, runtime={runtime}.")
        return runtime

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
        if memory is None:
            return 0

        if isinstance(memory, str):
            memory: int = memory_bytes(memory)

        if memory > 128 * 1024**3:
            err_msg = f"Invalid device memory provided, memory={memory / 1024**2} MB. Provide a value between 256 MB and 128 GB."
            logger.error(err_msg)
            raise ValueError(err_msg)
        return memory

    @validator("device")
    def _validate_device(cls, device: str) -> str:
        """Validate the device."""
        if device.startswith("nvidia-"):
            device = "gpu"  # for now, we re-map all nvidia devices to gpu
        if device not in ["auto", "cpu", "gpu"]:
            raise ValueError(f"Invalid device, device={device}.")
        return device

    @validator("device_memory")
    def _validate_device_memory(cls, device_memory: Union[int, str]) -> Union[int, Literal["auto"]]:
        """Validate the device memory."""
        if isinstance(device_memory, str) and device_memory != "auto":
            device_memory: int = memory_bytes(device_memory)

        if isinstance(device_memory, int) and device_memory > 128 * 1024**3:
            err_msg = f"Invalid device memory provided, device_memory={device_memory / 1024**2} MB. Provide a value between 256 MB and 128 GB."
            logger.error(err_msg)
            raise ValueError(err_msg)
        return device_memory


class ModelSpecMetadataCatalog:
    """Model specification catalog.

    This is a singleton class that maintains the metadata for all the models
    registered with the hub. The metadata includes the model resources (device/host memory etc),
    and the model profile (memory/cpu usage, batch size, input/output shapes etc).

    Usage:
        >>> catalog = ModelSpecMetadataCatalog.get()
        >>> metadata: ModelSpecMetadata = spec.metadata(method)
        >>> resources: ModelResources = metadata.resources
        >>> profile: Dict[str, Any] = metadata.profile

    """

    _instance: Optional["ModelSpecMetadataCatalog"] = None
    """Singleton instance."""

    _metadata_catalog: Dict[str, "ModelSpecMetadata"] = {}
    """Model specification metadata registry."""

    _resources_catalog: Dict[str, "ModelResources"] = {}
    """Model resources catalog."""

    _profile_catalog: Dict[str, Dict[str, Any]] = {}
    """Model metadata catalog of various additional profiling details."""

    @classmethod
    def get(cls) -> "ModelSpecMetadataCatalog":
        """Get the singleton instance."""
        if cls._instance is None:
            from nos.hub import Hub  # noqa: F401

            cls._instance = cls()
            try:
                Hub.get()  # force import models
                cls._instance.load_profile_catalog()
            except FileNotFoundError:
                logger.warning(f"Model metadata catalog not found, path={NOS_PROFILE_CATALOG_PATH}.")
        return cls._instance

    def __contains__(self, model_method_id: Any) -> bool:
        """Check if the model spec metadata is available."""
        return model_method_id in self._metadata_catalog

    def __getstate__(self) -> Dict[str, Any]:
        """Return the state of the object."""
        return {
            "_metadata_catalog": self._metadata_catalog,
            "_resources_catalog": self._resources_catalog,
            "_profile_catalog": self._profile_catalog,
        }

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Set the state of the object."""
        self._metadata_catalog = state["_metadata_catalog"]
        self._resources_catalog = state["_resources_catalog"]
        self._profile_catalog = state["_profile_catalog"]
        self._instance = self

    def __getitem__(self, model_method_id: Any) -> "ModelSpecMetadata":
        """Load the model spec metadata."""
        try:
            return self._metadata_catalog[model_method_id]
        except KeyError:
            raise KeyError(f"Unavailable model (id={model_method_id}).")

    def __setitem__(self, model_method_id: Any, metadata: "ModelSpecMetadata"):
        """Add the model spec metadata."""
        self._metadata_catalog[model_method_id] = metadata

    def load(self, model_method_id: Any) -> "ModelSpec":
        """Load the model spec metadata (identical to __getitem__)."""
        return self[model_method_id]

    def load_profile_catalog(self) -> "ModelSpecMetadataCatalog":
        """Load the model profiles from a JSON catalog."""
        import pandas as pd

        if not NOS_PROFILE_CATALOG_PATH.exists():
            raise FileNotFoundError(f"Model metadata catalog not found, path={NOS_PROFILE_CATALOG_PATH}.")

        # Read the catalog
        df = pd.read_json(str(NOS_PROFILE_CATALOG_PATH), orient="records")
        columns = df.columns
        # Check if the catalog is valid with the required columns
        for col in [
            "model_id",
            "method",
            "runtime",
            "device_name",
            "device_type",
            "device_index",
            "version",
            "prof.batch_size",
            "prof.shape",
            "prof.forward::memory_cpu::allocated",
        ]:
            if col not in columns:
                raise ValueError(f"Invalid model profile catalog, missing column={col}.")
        # Update the registry
        for _, row in df.iterrows():
            model_id, method = row["model_id"], row["method"]
            additional_kwargs = {}
            try:
                device_memory = (
                    math.ceil(row["prof.forward::memory_gpu::allocated"] / 1024**2 / 500) * 500 * 1024**2
                )
                additional_kwargs["device_memory"] = device_memory
            except Exception:
                logger.debug(f"Unable to parse device memory, model_id={model_id}, method={method}.")
            self._resources_catalog[f"{model_id}/{method}"] = ModelResources(
                runtime=row["runtime"], device=row["device_name"], **additional_kwargs
            )
            self._profile_catalog[f"{model_id}/{method}"] = row.to_dict()


@dataclass
class ModelSpecMetadata:
    """Model specification metadata."""

    id: str
    """Model identifier."""
    method: str
    """Model method name."""
    task: TaskType = None
    """Task type (e.g. image_embedding, image_generation, object_detection_2d, etc)."""

    def __repr__(self) -> str:
        return (
            f"""ModelSpecMetadata(id={self.id}, task={self.task}, method={self.method},\n"""
            f"""                  resources={self.resources})"""
        )

    @property
    def resources(self) -> Union[None, ModelResources]:
        """Return the model resources."""
        catalog = ModelSpecMetadataCatalog.get()
        try:
            return catalog._resources_catalog[f"{self.id}/{self.method}"]
        except KeyError:
            logger.debug(f"Model resources not found in catalog, id={self.id}, method={self.method}.")
            return None

    @property
    def profile(self) -> Dict[str, Any]:
        """Return the model profile."""
        catalog = ModelSpecMetadataCatalog.get()
        try:
            return catalog._profile_catalog[f"{self.id}/{self.method}"]
        except KeyError:
            logger.debug(f"Model metadata not found in catalog, id={self.id}, method={self.method}.")
            return {}


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

    def __repr__(self):
        return f"""ModelSpec(id={self.id}, methods=({', '.join(list(self.signature.keys()))}), tasks=({', '.join([str(self.task(m)) for m in self.signature])}))"""

    @validator("id", pre=True)
    def _validate_id(cls, id: str) -> str:
        """Validate the model identifier."""
        regex = re.compile(r"^[a-zA-Z0-9\/._-]+$")  # allow alphanumerics, `/`, `.`, `_`, and `-`
        if not regex.match(id):
            raise ValueError(
                f"Invalid model id, id={id} can only contain alphanumerics characters, `/`, `.`, `_`, and `-`"
            )
        return id

    @validator("signature", pre=True)
    def _validate_signature(
        cls, signature: Union[FunctionSignature, Dict[str, FunctionSignature]], **kwargs: Dict[str, Any]
    ) -> Dict[str, FunctionSignature]:
        """Validate the model signature / signatures.

        Checks that the model class `cls` has the function name attribute
        as defined in the signature `function_name`.

        Args:
            signature (Union[FunctionSignature, Dict[str, FunctionSignature]]): Model signature.
            **kwargs: Keyword arguments.
        Returns:
            Dict[str, FunctionSignature]: Model signature.
        """
        if isinstance(signature, (list, tuple)):
            raise TypeError(f"Invalid signature provided, signature={signature}.")
        if isinstance(signature, FunctionSignature):
            signature = {signature.method: signature}
        for method, sig in signature.items():
            if method != sig.method:
                raise ValueError(f"Invalid method name provided, method={method}, sig.method={sig.method}.")
            if sig and sig.func_or_cls:
                model_cls = sig.func_or_cls
                if sig.method and not hasattr(model_cls, sig.method):
                    raise ValueError(f"Model class {model_cls} does not have function {sig.method}.")
        return signature

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
            logger.debug(f"Model metadata not found, id={self.id}.")
            return None

    def metadata(self, method: str = None) -> ModelSpecMetadata:
        """Return the model spec metadata for a given method (or defaults to default method)."""
        if method is None:
            method = self.default_method
        catalog = ModelSpecMetadataCatalog.get()
        try:
            metadata: ModelSpecMetadata = catalog._metadata_catalog[f"{self.id}/{method}"]
        except KeyError:
            logger.debug(f"Model metadata not found in catalog, id={self.id}, method={method}.")
            return ModelSpecMetadata(id=self.id, method=method)
        return metadata

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

    def set_default_method(self, method: str) -> None:
        """Set the default method name."""
        if method not in self.signature:
            raise ValueError(f"Invalid method name provided, method={method}.")

        # Update the default method in the signature
        signature = {}
        signature[method] = self.signature.pop(method)
        signature.update(self.signature)
        self.signature = signature
        # Clear the cached properties to force re-computation
        self.__dict__.pop("default_method", None)
        self.__dict__.pop("default_signature", None)

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
            *init_args: Positional arguments.
            **init_kwargs: Keyword arguments.
        Returns:
            Any: Model instance.
        """
        sig: FunctionSignature = self.default_signature
        return sig.func_or_cls(*init_args, **init_kwargs)

    @classmethod
    def from_yaml(cls, filename: str) -> "ModelSpec":
        raise NotImplementedError()

    @classmethod
    def from_cls(
        cls,
        func_or_cls: Callable,
        init_args: Tuple[Any, ...] = (),
        init_kwargs: Dict[str, Any] = {},  # noqa: B006
        method: str = "__call__",
        runtime_env: RuntimeEnv = None,
        model_id: str = None,
        **kwargs: Any,
    ) -> "ModelSpec":
        """Wrap custom models/classes into a nos-compatible model spec.

        Args:
            func_or_cls (Callable): Model function or class. For now, only classes are supported.
            init_args (Tuple[Any, ...]): Initialization arguments.
            init_kwargs (Dict[str, Any]): Initialization keyword arguments.
            method (str): Method name to be executed.
            runtime_env (RuntimeEnv): Runtime environment specification.
            model_id (str): Optional model identifier.
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
        # the __call__ method as the first method.
        if method in all_methods:
            methods.insert(0, method)  # first method is the default method
        logger.debug(f"Registering methods [methods={methods}].")

        # Add function signature for each method
        model_id: str = func_or_cls.__name__ if model_id is None else model_id
        signature: Dict[str, FunctionSignature] = {}
        metadata: Dict[str, ModelSpecMetadata] = {}
        for method in methods:
            # Add the function signature
            sig = FunctionSignature(
                func_or_cls,
                method=method,
                init_args=init_args,
                init_kwargs=init_kwargs,
            )
            signature[method] = sig
            metadata[method] = ModelSpecMetadata(model_id, method, task=None)
            logger.debug(f"Added function signature [method={method}, signature={sig}].")

        # Build the model spec from the function signature
        spec = cls(
            model_id,
            signature=signature,
            metadata=metadata,
            runtime_env=runtime_env,
        )
        return spec

    def _to_proto(self) -> nos_service_pb2.GenericResponse:
        """Convert the model spec to proto."""
        spec = copy.deepcopy(self)
        # Note (spillai): We only serialize the input/output
        # signatures and method of the spec. Notably, the
        # `func_or_cls` attribute is not serialized to avoid
        # the dependency on torch and other server-side dependencies.
        for method in spec.signature:
            spec.signature[method].func_or_cls = None
            spec.signature[method].init_args = ()
            spec.signature[method].init_kwargs = {}
        return nos_service_pb2.GenericResponse(
            response_bytes=dumps(spec),
        )

    @staticmethod
    def _from_proto(minfo: nos_service_pb2.GenericResponse) -> "ModelSpec":
        """Convert the generic response back to the spec."""
        return loads(minfo.response_bytes)


@dataclass
class ModelDeploymentSpec:
    """Model deployment specification."""

    num_replicas: int = 1
    """Number of replicas."""
    resources: ModelResources = None
    """Model resources."""


@dataclass
class ModelServiceSpec:
    """Model service that captures the model, deployment and service specifications."""

    model: ModelSpec
    """Model specification."""
    deployment: ModelDeploymentSpec
    """Model deployment specification."""
    service: Any = None
    """Model service."""
