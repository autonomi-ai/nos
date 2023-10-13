from typing import List

import pytest

from nos.executors.ray import RayExecutor, RayJobExecutor
from nos.test.conftest import ray_executor  # noqa: F401


def test_ray_executor(ray_executor: RayExecutor):  # noqa: F811
    """Test ray executor singleton."""
    assert ray_executor.is_initialized(), "RayExecutor should be initialized."

    # Test singleton
    assert ray_executor is RayExecutor.get(), "RayExecutor instance should be a singleton."

    # Get raylet pid
    pid = ray_executor.pid
    assert pid is not None
    assert isinstance(pid, int)


@pytest.mark.skip(reason="TODO: Fix this test.")
def test_ray_job_executor(ray_executor: RayExecutor):  # noqa: F811
    """Test ray job executor."""
    job_manager: RayJobExecutor = ray_executor.jobs
    assert job_manager is not None

    jobs = job_manager.list()
    assert jobs is not None
    assert isinstance(jobs, list)
    assert len(jobs) >= 0

    for job_id in jobs:
        assert job_manager.info(job_id) is not None
        assert job_manager.status(job_id) is not None
        assert job_manager.logs(job_id) is not None
        assert job_manager.wait(job_id) is not None


def test_ray_load_spec_compatibility(ray_executor: RayExecutor):  # noqa: F811
    """Test hub.load_spec compatibility with RayExecutor"""
    import ray

    from nos import hub
    from nos.hub import ModelSpec

    models: List[str] = hub.list()
    assert len(models) > 0

    # Create actors for each model
    for model_id in models:
        spec: ModelSpec = hub.load_spec(model_id)
        assert spec is not None
        assert isinstance(spec, ModelSpec)

        # Create actor class
        actor_class = ray.remote(spec.default_signature.func_or_cls)
        # Create actor handle from actor class
        actor_handle = actor_class.remote(*spec.default_signature.init_args, **spec.default_signature.init_kwargs)
        assert actor_handle is not None
        del actor_handle
