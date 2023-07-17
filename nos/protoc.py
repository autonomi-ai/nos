import itertools
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List

from grpc_tools import protoc

from nos.constants import NOS_CACHE_DIR, NOS_PATH
from nos.logging import logger


PROTO_PATHS = [NOS_PATH / "proto"]


@dataclass
class DynamicProtobufCompiler:
    _instance = None
    """Singleton instance."""
    cache_dir = NOS_CACHE_DIR / "protobuf"
    """Cache directory for compiled protobuf modules."""

    def __init__(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Add cache dir to sys.path
        sys.path.append(str(self.cache_dir))

        # Compile all proto files from all paths
        logger.debug(f"Compiling protos [dirs={PROTO_PATHS}]")
        for path in itertools.chain.from_iterable(Path(path).glob("*.proto") for path in PROTO_PATHS):
            logger.debug(f"Compiling ... [filename={path}]")
            self.compile(str(path))
        logger.debug(f"Compiled modules [modules={self.list_modules()}]")

    @classmethod
    def get(cls: "DynamicProtobufCompiler") -> "DynamicProtobufCompiler":
        """Get DynamicProtobufCompiler."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def compile(self, proto_filename: str):
        """Compile the proto file to generate the Python modules"""
        logger.debug(f"Compiling proto [proto={proto_filename}]")
        cmd = [
            "",
            "-I" + str(Path(protoc.__file__).parent / "_proto/"),
            f"--python_out={self.cache_dir}",
            f"--grpc_python_out={self.cache_dir}",
            f"--proto_path={Path(proto_filename).parent}",
            f"{Path(proto_filename).name}",
        ]
        logger.debug(f"Compiling proto [cmd=protoc {' '.join(cmd)}]")

        st = time.time()
        protoc.main(cmd)
        logger.debug(f"Compilation done [elapsed={(time.time() - st)*1e3:.1f}ms]")

    def list_modules(self) -> List[str]:
        """Return a list of compiled modules."""
        return [Path(path).stem for path in self.cache_dir.glob("*_pb2*.py")]

    def import_module(self, module_name: str):
        """Import the specified module and return the imported module object."""
        import importlib.util

        # Load the module
        module_path = f"{Path(self.cache_dir) / module_name}.py"
        logger.debug(f"Loading module [module={module_path}]")
        spec = importlib.util.spec_from_file_location(module_name, module_path)

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


@lru_cache(maxsize=None)
def import_module(module_name: str):
    """Import the specified module and return the imported module object."""
    compiler = DynamicProtobufCompiler.get()
    return compiler.import_module(module_name)
