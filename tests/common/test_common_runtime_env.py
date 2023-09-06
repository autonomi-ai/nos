from nos.common.runtime_env import RuntimeEnv, RuntimeEnvironmentsHub, register


def test_runtime_env():
    """Test the runtime environment."""

    TEST_ENV = "onnx-rt"

    # Create a runtime environment
    runtime_env = RuntimeEnv.from_packages(["onnx"])
    assert runtime_env is not None

    # Register the runtime environment
    register(TEST_ENV, runtime_env)

    # Check if the runtime environment is registered
    assert TEST_ENV in RuntimeEnvironmentsHub.list()
    assert RuntimeEnvironmentsHub.get(TEST_ENV) is not None
    assert RuntimeEnvironmentsHub[TEST_ENV] is not None
