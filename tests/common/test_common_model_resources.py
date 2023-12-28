import pytest

from nos.common.spec import ModelResources


@pytest.fixture
def model_resources():
    return ModelResources()


def test_model_resources_default_values(model_resources):
    assert model_resources.runtime == "auto"
    assert model_resources.device == "auto"
    assert model_resources.cpus == 0
    assert model_resources.memory == 0
    assert model_resources.device_memory == "auto"


def test_model_resources_validate_runtime():
    from nos.server._runtime import InferenceServiceRuntime

    for runtime in list(InferenceServiceRuntime.configs.keys()) + ["auto"]:
        resources = ModelResources(runtime=runtime)
        assert resources is not None

    with pytest.raises(ValueError):
        ModelResources(runtime="invalid_runtime")


def test_model_resources_validate_cpus():
    with pytest.raises(ValueError):
        ModelResources(cpus=-1)
    with pytest.raises(ValueError):
        ModelResources(cpus=129)


def test_model_resources_validate_memory():
    with pytest.raises(ValueError):
        ModelResources(memory="invalid_memory")


def test_model_resources_validate_device_memory():
    with pytest.raises(ValueError):
        ModelResources(device_memory="invalid_device_memory")


def test_model_resources_string_formatting():
    from nos.common import ModelResources

    resources = ModelResources(cpus=1, memory="100Mi")
    assert resources is not None
    assert resources.cpus == 1
    assert resources.memory == 100 * 1024 * 1024

    resources = ModelResources(cpus=1, memory="1Gi")
    assert resources is not None
    assert resources.memory == 1024**3
