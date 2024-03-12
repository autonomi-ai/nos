"""Client-side integration tests

Requirements:
- test in client-only environment (pip install torch-nos)

Benchmarks:
- See benchmark.md
"""


import pytest


# Skip this entire test if server is not installed
pytestmark = pytest.mark.skipif(pytest.importorskip("ray") is not None, reason="ray is not installed")


def test_client_only_installation():
    # Ensure that the environment does not have server-side requirements installed
    try:
        import ray  # noqa: F401
        import torch  # noqa: F401

        raise AssertionError("torch/ray is installed in pixeltable environment")
    except ImportError:
        pass

    # Try to import nos (should work fine in client-only environment)
    import nos  # noqa: F401
