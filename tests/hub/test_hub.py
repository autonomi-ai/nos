import pytest

from nos import hub
from nos.hub import ModelSpec
from nos.test.utils import PyTestGroup, requires_torch_cuda


def test_hub_list():
    model_names = hub.list()
    assert len(model_names) > 0


@requires_torch_cuda
def test_hub_load():
    models = hub.list()
    assert len(models) > 0

    # Load the model with the first model name
    model = hub.load(models[0])
    assert model is not None

    # Check if hub.load raises an error when the
    # model name is invalid
    with pytest.raises(KeyError):
        hub.load("not-a-model-name")


@requires_torch_cuda
def test_hub_load_spec():
    models = hub.list()
    assert len(models) > 0

    # Load the model spec with the first model name
    spec = hub.load_spec(models[0])
    assert spec is not None
    assert isinstance(spec, ModelSpec)

    # Check if hub.load_spec raises an error when the
    # model name is invalid
    with pytest.raises(KeyError):
        hub.load_spec("not-a-model-name")


@requires_torch_cuda
@pytest.mark.benchmark(group=PyTestGroup.BENCHMARK_MODELS)
def test_hub_load_all():
    models = hub.list()
    assert len(models) > 0

    for model_name in models:
        model = hub.load(model_name)
        assert model is not None
