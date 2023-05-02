import pytest

from nos.executors.ray import RayExecutor


@pytest.mark.e2e
def test_ray_executor():
    """Test ray executor singleton."""
    # Test singleton
    executor = RayExecutor.get()
    executor_ = RayExecutor.get()
    assert executor is executor_

    # Check if Ray is initialized
    assert not executor.is_initialized()
    executor_ = RayExecutor.get()
    assert executor is executor_

    # Initialize Ray executor
    success = executor.init()
    assert success
    assert executor.is_initialized()

    # Get raylet pid
    pid = executor.pid
    assert pid is not None
    assert isinstance(pid, int)

    # Stop Ray executor
    executor.stop()
    pid = executor.pid
    assert pid is None


@pytest.mark.skip(reason="Not yet implemented.")
def test_ray_hub_compatibility():
    """Test hub.load_spec compatibility with RayExecutor"""
    import ray

    from nos import hub
    from nos.hub import ModelSpec

    models = hub.list()
    assert len(models) > 0

    # Create actors for each model
    for model_name in models:
        spec = hub.load_spec(model_name)
        assert spec is not None
        assert isinstance(spec, ModelSpec)

        # Create actor class
        actor_class = ray.remote(spec.cls)
        # Create actor handle from actor class
        actor_handle = actor_class.remote(*spec.args, **spec.kwargs)
        assert actor_handle is not None
        del actor_handle
