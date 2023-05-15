import pytest


@pytest.mark.skip(reason="Not yet implemented.")
def test_serve_simple_client():
    from nos.serve import NOS_SERVE_DEFAULT_HTTP_HOST, NOS_SERVE_DEFAULT_HTTP_PORT
    from nos.serve.client import SimpleClient

    client = SimpleClient(NOS_SERVE_DEFAULT_HTTP_HOST, NOS_SERVE_DEFAULT_HTTP_PORT)
    assert client is not None

    # TODO (spillai): Test is_healthy(), wait(), and post() methods
