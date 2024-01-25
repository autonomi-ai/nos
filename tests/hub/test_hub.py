from typing import List

import pytest

from nos import hub
from nos.common import ModelSpec, ModelSpecMetadata, TaskType
from nos.hub import Hub
from nos.logging import logger
from nos.test.utils import PyTestGroup, skip_if_no_torch_cuda


def test_hub_singleton_instance():
    """Test the singleton instance of the hub."""
    hub = Hub.get()
    assert hub is Hub.get(), "Hub instance should be a singleton."


def test_hub_list():
    """Test listing all models from the hub."""
    model_names: List[str] = hub.list()
    assert model_names is not None
    assert isinstance(model_names, list)
    assert isinstance(model_names[0], str)
    assert len(model_names) > 0, "No models found in the registry."

    for model_name in model_names:
        hub_instance: Hub = Hub.get()
        assert model_name in hub_instance


def test_hub_load_spec_all():
    """Load all model specs from the hub."""

    models: List[str] = hub.list()
    assert len(models) > 0

    for model_id in models:
        assert model_id is not None
        assert isinstance(model_id, str)

        spec: ModelSpec = hub.load_spec(model_id)
        assert spec is not None

        # Check if all hub registered models have valid metadata / tasks
        for method in spec.signature:
            md: ModelSpecMetadata = spec.metadata(method)
            assert md is not None, f"Model spec (id={model_id}) should have metadata for method={method}."
            assert isinstance(md, ModelSpecMetadata)

            task: TaskType = spec.task(method)
            assert task is not None
            assert isinstance(task, TaskType)
            logger.debug(f"Model [id={model_id}, task={task}, method={method}]")

        assert spec.task is not None

    # Check if hub.load_spec raises an error when the
    # model name is invalid
    with pytest.raises(Exception):
        hub.load_spec("not-a-model-name")


# Note (spillai): This test is a good substitute for the
# test_hub_load_all test below since that requires a GPU.
def test_hub_load_noop():
    """Load the noop model from the hub."""
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


def test_hub_catalog():
    """Test loading models from the hub catalog."""
    import os

    from nos.test.utils import NOS_TEST_DATA_DIR

    pvalue = os.getenv("NOS_HUB_CATALOG_PATH", "")
    nmodels = len(Hub.list())
    os.environ["NOS_HUB_CATALOG_PATH"] = ":".join(
        [
            str(yaml)
            for yaml in [
                NOS_TEST_DATA_DIR / "hub/custom_model/config.yaml",
                NOS_TEST_DATA_DIR / "hub/custom_model/config-with-replicas.yaml",
            ]
        ]
    )
    Hub.register_from_catalog()
    os.environ["NOS_HUB_CATALOG_PATH"] = pvalue
    assert (
        len(Hub.list()) >= nmodels + 1
    ), "Failed to register custom model from catalog, assumes at least one model is registered."


def test_hub_register_from_yaml():
    """Test loading models from the hub yaml config."""
    from nos.test.utils import NOS_TEST_DATA_DIR

    # Test valid model configs
    for yaml in [
        NOS_TEST_DATA_DIR / "hub/custom_model/config.yaml",
        NOS_TEST_DATA_DIR / "hub/custom_model/config-alternate-init-kwargs.yaml",
        NOS_TEST_DATA_DIR / "hub/custom_model/config-with-existing-model-id.yaml",
    ]:
        Hub.register_from_yaml(yaml)

    # Disable logging for following tests
    logger.disable("nos")

    # Test invalid/malformed model configs
    with pytest.raises(FileNotFoundError):
        Hub.register_from_yaml(NOS_TEST_DATA_DIR / "hub/custom_model/config-not-found.yaml")

    for yaml in [
        NOS_TEST_DATA_DIR / "hub/custom_model/config-malformed-model-cls.yaml",
        NOS_TEST_DATA_DIR / "hub/custom_model/config-malformed-model-path.yaml",
        NOS_TEST_DATA_DIR / "hub/custom_model/config-malformed-model-method.yaml",
    ]:
        logger.debug(f"Loading model spec from malformed YAML: {yaml}")
        with pytest.raises(Exception):
            Hub.register_from_yaml(yaml)

    logger.enable("nos")


@skip_if_no_torch_cuda
@pytest.mark.benchmark(group=PyTestGroup.MODEL_BENCHMARK)
def test_hub_load_all():
    """Benchmark loading all models from the hub."""
    models: List[str] = hub.list()
    assert len(models) > 0

    for model_id in models:
        model = hub.load(model_id)
        assert model is not None
