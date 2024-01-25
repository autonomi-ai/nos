import os
from pathlib import Path


NOS_HOME = Path(os.getenv("NOS_HOME", str(Path.home() / ".nos")))
NOS_CACHE_DIR = NOS_HOME / "cache"
NOS_MODELS_DIR = NOS_HOME / "models"
NOS_LOG_DIR = NOS_HOME / "logs"
NOS_TMP_DIR = NOS_HOME / "tmp"
NOS_PATH = Path(__file__).parent

NOS_HOME.mkdir(parents=True, exist_ok=True)
NOS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
NOS_MODELS_DIR.mkdir(parents=True, exist_ok=True)
NOS_LOG_DIR.mkdir(parents=True, exist_ok=True)
NOS_TMP_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_HTTP_HOST = os.getenv("NOS_HTTP_HOST", "127.0.0.1")
DEFAULT_HTTP_PORT = int(os.getenv("NOS_HTTP_PORT", 8000))
DEFAULT_HTTP_ADDRESS = f"{DEFAULT_HTTP_HOST}:{DEFAULT_HTTP_PORT}"

DEFAULT_GRPC_HOST = os.getenv("NOS_GRPC_HOST", "[::]")
DEFAULT_GRPC_PORT = int(os.getenv("NOS_GRPC_PORT", 50051))
DEFAULT_GRPC_ADDRESS = f"{DEFAULT_GRPC_HOST}:{DEFAULT_GRPC_PORT}"

GRPC_MAX_MESSAGE_LENGTH = 32 * 1024 * 1024  # 32 MB
GRPC_MAX_WORKER_THREADS = int(os.getenv("NOS_GRPC_MAX_WORKER_THREADS", 4))

NOS_PROFILING_ENABLED = bool(int(os.getenv("NOS_PROFILING_ENABLED", "0")))
NOS_MEMRAY_ENABLED = bool(int(os.getenv("NOS_MEMRAY_ENABLED", "0")))

NOS_RAY_NS = os.getenv("NOS_RAY_NS", "nos-dev")
NOS_RAY_ENV = os.environ.get("NOS_ENV", os.getenv("CONDA_DEFAULT_ENV", None))
NOS_RAY_DASHBOARD_ENABLED = bool(int(os.getenv("NOS_RAY_DASHBOARD_ENABLED", "0")))
NOS_RAY_OBJECT_STORE_MEMORY = int(os.getenv("NOS_RAY_OBJECT_STORE_MEMORY", 2 * 1024 * 1024 * 1024))  # 2GB
NOS_RAY_JOB_CLIENT_ADDRESS = "http://127.0.0.1:8265"

NOS_PROFILE_CATALOG_PATH = NOS_PATH / "catalogs/model_profile_catalog.json"
