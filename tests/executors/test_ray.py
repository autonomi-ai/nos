from nos.executors.ray import RayExecutor
from nos.test.conftest import ray_executor  # noqa: F401


def test_ray_executor(ray_executor: RayExecutor):  # noqa: F811
    """Test ray executor singleton."""
    assert ray_executor.is_initialized()

    # Test singleton
    ray_executor_ = RayExecutor.get()
    assert ray_executor is ray_executor_

    # Get raylet pid
    pid = ray_executor.pid
    assert pid is not None
    assert isinstance(pid, int)


def test_ray_load_spec_compatibility(ray_executor: RayExecutor):  # noqa: F811
    """Test hub.load_spec compatibility with RayExecutor"""
    import ray

    from nos import hub
    from nos.hub import ModelSpec

    models = hub.list()
    assert len(models) > 0

    # Create actors for each model
    for spec in models:
        assert spec is not None
        assert isinstance(spec, ModelSpec)

        # Create actor class
        actor_class = ray.remote(spec.signature.func_or_cls)
        # Create actor handle from actor class
        actor_handle = actor_class.remote(*spec.signature.init_args, **spec.signature.init_kwargs)
        assert actor_handle is not None
        del actor_handle
