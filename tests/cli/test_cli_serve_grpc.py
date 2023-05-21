import pytest
from typer.testing import CliRunner

from nos.cli.cli import app_cli
from nos.test.conftest import grpc_server_docker_runtime_cpu  # noqa: F401
from nos.test.utils import NOS_TEST_IMAGE


runner = CliRunner()


@pytest.mark.e2e
def test_cli_serve_grpc_list(grpc_server_docker_runtime_cpu):  # noqa: F811
    from nos.test.conftest import GRPC_TEST_PORT_CPU

    result = runner.invoke(app_cli, ["serve-grpc", "-a", f"localhost:{GRPC_TEST_PORT_CPU}", "list"])
    assert result.exit_code == 0


@pytest.mark.e2e
def test_cli_serve_grpc_predict_img2vec(grpc_server_docker_runtime_cpu):  # noqa: F811
    from nos.test.conftest import GRPC_TEST_PORT_CPU

    result = runner.invoke(
        app_cli, ["serve-grpc", "-a", f"localhost:{GRPC_TEST_PORT_CPU}", "img2vec", "-i", NOS_TEST_IMAGE]
    )
    assert result.exit_code == 0
