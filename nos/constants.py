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

DEFAULT_GRPC_PORT = 50051
DEFAULT_HTTP_PORT = 8000

NOS_PROFILING_ENABLED = bool(int(os.getenv("NOS_PROFILING_ENABLED", "0")))
NOS_DASHBOARD_ENABLED = bool(int(os.getenv("NOS_DASHBOARD_ENABLED", "0")))
NOS_MEMRAY_ENABLED = bool(int(os.getenv("NOS_MEMRAY_ENABLED", "0")))
