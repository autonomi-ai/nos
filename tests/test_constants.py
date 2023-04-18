from nos.constants import NOS_CACHE_DIR, NOS_HOME, NOS_MODELS_DIR


def test_nos_home():
    assert NOS_HOME.exists()
    assert NOS_HOME.is_dir()


def test_nos_cache_dir():
    assert NOS_CACHE_DIR.exists()
    assert NOS_CACHE_DIR.is_dir()


def test_nos_models_dir():
    assert NOS_MODELS_DIR.exists()
    assert NOS_MODELS_DIR.is_dir()
