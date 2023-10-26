import pytest


def test_nos_imports():
    pass


@pytest.mark.skip
def test_nos_internal_imports():
    import nos

    # Try importing the internal module
    try:
        import nos._internal  # noqa: F401, F403

        success = True
    except ImportError:
        success = False

    assert success == nos.internal_libs_available()
