import inspect
from dataclasses import field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import yaml
from pydantic import ValidationError, root_validator
from pydantic.dataclasses import dataclass
from pydantic.errors import ConfigError

from nos.common.metaclass import SingletonMetaclass  # noqa: F401
from nos.common.spec import (  # noqa: F401
    FunctionSignature,
    ModelSpec,
    ModelSpecMetadata,
    ModelSpecMetadataRegistry,
    TaskType,
)
from nos.hub.config import HuggingFaceHubConfig, NosHubConfig, TorchHubConfig  # noqa: F401
from nos.hub.hf import hf_login  # noqa: F401
from nos.logging import logger


class Hub:
    """Registry for models."""

    _instance: Optional["Hub"] = None
    """Singleton instance."""
    _registry: Dict[str, ModelSpec] = {}
    """Model specifications lookup for all models registered."""

    @classmethod
    def get(cls: "Hub") -> "Hub":
        """Get the singleton instance.

        Returns:
            Hub: Singleton instance.
        """
        if cls._instance is None:
            cls._instance = cls()
            # Register models / Populate the registry
            import nos.models  # noqa: F401, E402

            # Register models from the catalog dynamically
            cls.register_from_catalog()
        return cls._instance

    @classmethod
    def list(cls, private: bool = False) -> List[str]:
        """List models in the registry.

        Args:
            private (bool): Whether to include private models.

        Returns:
            List[str]: List of model names.
        """
        if private:
            raise NotImplementedError("Private models not supported.")
        return [k for k in cls.get()._registry.keys()]

    @classmethod
    def load_spec(cls, model_id: str) -> ModelSpec:
        """Load model spec from the registry.

        Args:
            model_id (str): Model identifier (e.g. `openai/clip-vit-base-patch32`).

        Returns:
            ModelSpec: Model specification.
        """
        try:
            return cls.get()._registry[model_id]
        except KeyError:
            raise KeyError(f"Unavailable model (name={model_id}).")

    @classmethod
    def load(cls, model_id: str) -> Any:
        """Instantiate model from the registry.

        Args:
            model_id (str): Model identifier (e.g. `openai/clip-vit-base-patch32`).

        Returns:
            Any: Instantiated model.
        """
        spec: ModelSpec = cls.load_spec(model_id)
        # Note (spillai): Loading the default signature here is OK
        # since all the signatures have the same `func_or_cls`.
        sig: FunctionSignature = spec.default_signature
        return sig.func_or_cls(*sig.init_args, **sig.init_kwargs)

    @classmethod
    def register(
        cls,
        model_id: str,
        task: TaskType,
        func_or_cls: Callable,
        method: str = "__call__",
        init_args: Tuple[Any] = (),
        init_kwargs: Dict[str, Any] = {},  # noqa: B006
        inputs: Dict[str, Any] = {},  # noqa: B006
        outputs: Union[Any, Dict[str, Any], None] = None,  # noqa: B006
        **kwargs,
    ) -> ModelSpec:
        """Model registry decorator.

        Args:
            model_id (str): Model identifier (e.g. `openai/clip-vit-base-patch32`).
            task (TaskType): Task type (e.g. `TaskType.OBJECT_DETECTION_2D`).
            func_or_cls (Type[Any]): Model function or class.
            **kwargs: Additional keyword arguments.
        Returns:
            ModelSpec: Model specification.
        """
        logger.debug(
            f"""Registering model [model={model_id}, task={task}, func_or_cls={func_or_cls}, """
            f"""inputs={inputs}, outputs={outputs}, """
            f"""init_args={init_args}, init_kwargs={init_kwargs}, method={method}]"""
        )

        # Create signature
        signature: Dict[str, FunctionSignature] = {
            method: FunctionSignature(
                func_or_cls,
                method=method,
                init_args=init_args,
                init_kwargs=init_kwargs,
                input_annotations=inputs,
                output_annotations=outputs,
            ),
        }
        # Add metadata for the model
        metadata: Dict[str, ModelSpecMetadata] = {
            method: ModelSpecMetadata(model_id, method, task),
        }
        spec = ModelSpec(model_id, signature=signature, _metadata=metadata)
        logger.debug(f"Created model spec [id={model_id}, spec={spec}, metadata={metadata}]")

        # Get hub instance
        hub = cls.get()

        # Register model id to model spec registry
        if model_id not in hub._registry:
            hub._registry[model_id] = spec
            logger.debug(f"Registered model to hub registry [id={model_id}, spec={spec}]")

        # Add another signature if the model is already registered
        else:
            _spec = hub._registry[model_id]
            if method not in _spec.signature:
                logger.debug(
                    f"Adding task signature [model={model_id}, task={task}, method={method}, sig={spec.signature}]"
                )
                _spec.signature[method] = spec.signature[method]
                _spec._metadata[method] = spec._metadata[method]
            else:
                logger.debug(
                    f"Task signature already registered [model={model_id}, task={task}, method={method}, sig={spec.signature}]"
                )

        logger.debug(f"Registered model [id={model_id}, spec={spec}]")
        return spec

    @classmethod
    def register_from_yaml(cls, filename: str) -> List["ModelSpec"]:
        """Register models from a catalog YAML.

        Args:
            filename (str): Path to the deployment config YAML file.
        Returns:
            List[ModelSpec]: List of model specifications.
        """

        @dataclass
        class _ModelImportConfig:
            """Model import configuration."""

            id: str
            """Model identifier."""
            model_path: str
            """Model path."""
            model_cls: Callable
            """Model class name."""
            default_method: str
            """Default model method name."""
            runtime_env: str
            """Runtime environment."""

            @root_validator(pre=True, allow_reuse=True)
            def _validate_model_cls_import(cls, values):
                """Validate the model."""
                import importlib
                import importlib.util
                import sys

                model_cls_name = values.get("model_cls")
                model_path = values.get("model_path")

                # Check if model_path is a valid path relative to the directory containing
                # the catalog.yaml file.
                model_path = Path(filename).parent / model_path
                if not model_path.exists():
                    logger.error(f"Invalid model path provided, model_path={model_path}.")
                    raise FileNotFoundError(f"Invalid model path provided, model_path={model_path}.")

                # Check if the model_cls is importable from the model_path
                try:
                    # Load `model_cls` from the `model_path`
                    logger.debug(f"Loading model class [model_cls={model_cls_name}, model_path={model_path}].")
                    sys.path.append(str(model_path.parent))
                    logger.debug(f"Added model path to sys.path [model_path={model_path.parent}].")
                    spec = importlib.util.spec_from_file_location(model_cls_name, model_path)
                    logger.debug(f"Loaded spec from file location [spec={spec}].")
                    module = importlib.util.module_from_spec(spec)
                    logger.debug(f"Loaded module from spec [module={module}].")
                    spec.loader.exec_module(module)
                    logger.debug(f"Executed module [module={module}].")
                    model_cls = getattr(module, model_cls_name)
                    logger.debug(f"Loaded model class [model_cls={model_cls}].")
                except Exception as e:
                    import traceback

                    tback_str = traceback.format_exc()
                    logger.error(
                        f"Failed to import model class, model_cls={model_cls_name}, model_path={model_path}, e={e}\n\n{tback_str}."
                    )
                    raise ValueError(
                        f"Invalid model class provided, model_cls={model_cls_name}, model_path={model_path}, e={e}\n\n{tback_str}."
                    )

                values.update(model_cls=model_cls)
                return values

        # Check if the file exists and has a YAML extension
        path = Path(filename)
        logger.debug(f"Loading deployment configuration from {path}")
        if not path.exists():
            raise FileNotFoundError(f"YAML file {path.absolute()} does not exist")
        if not (path.name.endswith(".yaml") or path.name.endswith(".yml")):
            raise ValueError(f"YAML file {path.absolute()} must have a .yaml or .yml extension")

        # Load the YAML file
        with path.open("r") as f:
            data = yaml.safe_load(f)
        if "models" not in data:
            raise ValueError("Missing `models` specification in the YAML file")

        # Register the models
        specs: List[ModelSpec] = []
        for model_id, mconfig in data["models"].items():
            # Add the model id to the config
            mconfig.update({"id": model_id})
            # Generate the model spec from the config
            try:
                mconfig = _ModelImportConfig(**mconfig)
            except (ValidationError, ConfigError) as e:
                raise ValueError(f"Invalid model config provided, filename={filename}, e={e}")

            # Register the model as a custom model
            spec: ModelSpec = cls.register(
                mconfig.id,
                TaskType.CUSTOM,
                mconfig.model_cls,
                method=mconfig.default_method,
            )
            logger.debug(f"Registered model [id={model_id}, spec={spec}]")
            specs.append(spec)
        return specs

    @classmethod
    def register_from_catalog(cls):
        """Register models from the catalog.

        The current workflow for registering models is as follows:
         - Load all .yaml files from the environment variable `NOS_HUB_CATALOG_PATH`
         - Register models from each of the catalog files (via `register_from_yaml`)

            `register_from_catalog` -> `register_from_yaml` -> `register`

        Raises:
            TypeError: If `NOS_HUB_CATALOG_PATH` is not a string.
            FileNotFoundError: If a specified catalog file does not exist.
        """
        import os

        # from nos.common.config import DeploymentConfig

        NOS_HUB_CATALOG_PATH = os.getenv("NOS_HUB_CATALOG_PATH", "")
        if not isinstance(NOS_HUB_CATALOG_PATH, str):
            raise TypeError(f"NOS_HUB_CATALOG_PATH must be a string, got {type(NOS_HUB_CATALOG_PATH)}")

        logger.debug("Loading hub models from catalog.")
        paths: List[Path] = [
            Path(filename)
            for filename in NOS_HUB_CATALOG_PATH.split(":")
            if filename.endswith(".yaml") or filename.endswith(".yml")
        ]
        logger.debug(f"Found {len(paths)} catalog files.")

        specs = []
        for path in paths:
            logger.debug(f"Loading catalog file {path}.")
            if not path.exists():
                raise FileNotFoundError(f"Catalog file {path} does not exist.")
            _specs = cls.register_from_yaml(str(path))
            specs.extend(_specs)
        logger.debug(f"Registered {len(specs)} models from catalog.")


# Alias methods
list = Hub.list
load = Hub.load
register = Hub.register
load_spec = Hub.load_spec
