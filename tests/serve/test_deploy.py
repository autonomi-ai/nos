import pytest  # noqa: F401

from nos.test.utils import requires_torch_cuda


@pytest.mark.benchmark
@requires_torch_cuda
def test_deployment():
    from threading import Thread

    from nos import hub, serve
    from nos.serve.client import SimpleClient

    # Create a deployment from the model handle
    model_name = "stabilityai/stable-diffusion-2"
    model_spec = hub.load_spec(model_name)

    # Create a deployment from the model handle
    def deploy():
        serve.deployment(
            model_name,
            model_spec,
            deployment_config={
                "ray_actor_options": {"num_gpus": 1},
                "autoscaling_config": {"min_replicas": 0, "max_replicas": 2},
            },
            daemon=True,
        )

    t = Thread(target=deploy)
    t.start()

    # Check the health of the deployment
    client = SimpleClient()

    # Wait for the deployment to be ready (30 seconds timeout)
    # Raises TimeoutError if the deployment is not ready
    client.wait(timeout=30)

    # # Post to the deployment
    # print("Posting to the deployment...")
    # response = client.post("predict", json={"prompt": "a cat dancing on the grass."})
    # assert response.status_code == 201

    # Join the thread
    print("Joining the thread...")
    t.join()
