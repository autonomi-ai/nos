import pytest
from typer.testing import CliRunner

from nos.cli.cli import app_cli


pytestmark = pytest.mark.client
runner = CliRunner()


def test_cli_system_info():
    result = runner.invoke(app_cli, ["system", "info"])
    assert result.exit_code == 0
