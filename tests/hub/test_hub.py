from typing import List

import pytest

from nos import hub
from nos.hub import ModelSpec
from nos.test.utils import PyTestGroup, skip_if_no_torch_cuda


def test_hub_singleton_instance():
    from nos.hub import Hub

    hub = Hub.get()
    assert hub is Hub.get(), "Hub instance should be a singleton."


def test_hub_list():
    model_names: List[str] = hub.list()
    assert len(model_names) > 0, "No models found in the registry."


def test_hub_load_spec_all():
    models: List[str] = hub.list()
    assert len(models) > 0

    for model_id in models:
        assert model_id is not None
        assert isinstance(model_id, str)

    # Check if hub.load_spec raises an error when the
    # model name is invalid
    with pytest.raises(Exception):
        hub.load_spec("not-a-model-name")


@skip_if_no_torch_cuda
def test_hub_load():
    models: List[str] = hub.list()
    assert len(models) > 0

    # Load the model with the first model name
    spec: ModelSpec = hub.load_spec("noop/process-images")
    assert isinstance(spec, ModelSpec)

    # Loads the model with the first model name
    model: ModelSpec = hub.load("noop/process-images")
    assert model is not None

    # Check if hub.load raises an error when the
    # model name is invalid
    with pytest.raises(Exception):
        hub.load("not-a-model-name")


@skip_if_no_torch_cuda
@pytest.mark.benchmark(group=PyTestGroup.MODEL_BENCHMARK)
def test_hub_load_all():
    """Benchmark loading all models from the hub."""
    models: List[str] = hub.list()
    assert len(models) > 0

    for model_id in models:
        model = hub.load(model_id)
        assert model is not None
