import os
from pathlib import Path


NOS_HOME = Path(os.getenv("NOS_HOME", str(Path.home() / ".nos")))
NOS_CACHE_DIR = NOS_HOME / "cache"
NOS_MODELS_DIR = NOS_HOME / "models"

NOS_HOME.mkdir(parents=True, exist_ok=True)
NOS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
NOS_MODELS_DIR.mkdir(parents=True, exist_ok=True)
