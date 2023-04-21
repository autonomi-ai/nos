import pytest

from nos import hub
from nos.test.utils import PyTestGroup, requires_torch_cuda


def test_hub_list():
    model_names = hub.list()
    assert len(model_names) > 0


@requires_torch_cuda
def test_hub_load():
    models = hub.list()
    assert len(models) > 0

    model = hub.load(models[0])
    assert model is not None


@requires_torch_cuda
@pytest.mark.benchmark(group=PyTestGroup.BENCHMARK_MODELS)
def test_hub_load_all():
    models = hub.list()
    assert len(models) > 0

    for model_name in models:
        model = hub.load(model_name)
        assert model is not None
