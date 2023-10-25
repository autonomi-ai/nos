import pytest
from typer.testing import CliRunner


pytestmark = pytest.mark.cli
runner = CliRunner()


def test_cli_system_help():
    from nos.cli.cli import app_cli

    result = runner.invoke(app_cli, ["system", "--help"])
    assert result.exit_code == 0


def test_cli_system_info():
    from nos.cli.cli import app_cli

    result = runner.invoke(app_cli, ["system", "info"])
    assert result.exit_code == 0
