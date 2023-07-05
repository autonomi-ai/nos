import pytest


@pytest.mark.client
@pytest.mark.parametrize("runtime", ["cpu", "gpu", "auto"])
def test_nos_init(runtime):  # noqa: F811
    """Test the NOS server daemon initialization."""
    import nos
    from nos.client import InferenceClient
    from nos.server.runtime import InferenceServiceRuntime

    # Initialize the server
    GRPC_PORT = 50055
    container = nos.init(runtime=runtime, port=GRPC_PORT, utilization=0.5)
    assert container is not None
    assert container.id is not None
    containers = InferenceServiceRuntime.list()
    assert len(containers) == 1

    # Test waiting for server to start
    # This call should be instantaneous as the server is already ready for the test
    client = InferenceClient(f"[::]:{GRPC_PORT}")
    assert client.WaitForServer(timeout=180, retry_interval=5)
    assert client.IsHealthy()

    # Test re-initializing the server
    container_ = nos.init(runtime=runtime, port=GRPC_PORT, utilization=0.5)
    assert container_.id == container.id

    # Shutdown the server
    nos.shutdown()
    containers = InferenceServiceRuntime.list()
    assert len(containers) == 0
