import pytest

from nos import hub
from nos.hub import ModelSpec
from nos.test.utils import PyTestGroup, skip_if_no_torch_cuda


def test_hub_list():
    model_names = hub.list()
    assert len(model_names) > 0


def test_hub_load_spec_all():
    models = hub.list()
    assert len(models) > 0

    for spec in models:
        assert spec is not None
        assert isinstance(spec, ModelSpec)

    # Check if hub.load_spec raises an error when the
    # model name is invalid
    with pytest.raises(Exception):
        hub.load_spec("not-a-model-name")


@skip_if_no_torch_cuda
def test_hub_load():
    models = hub.list()
    assert len(models) > 0

    # Load the model with the first model name
    model = hub.load(models[0].name, task=models[0].task)
    assert model is not None

    # Check if hub.load raises an error when the
    # model name is invalid
    with pytest.raises(KeyError):
        hub.load("not-a-model-name")


@skip_if_no_torch_cuda
@pytest.mark.benchmark(group=PyTestGroup.MODEL_BENCHMARK)
def test_hub_load_all():
    """Benchmark loading all models from the hub."""
    models = hub.list()
    assert len(models) > 0

    for spec in models:
        model = hub.load(spec.name, task=spec.task)
        assert model is not None
