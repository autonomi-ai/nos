from typer.testing import CliRunner

from nos.cli.cli import app_cli


runner = CliRunner()


def test_system_info():
    result = runner.invoke(app_cli, ["system", "info"])
    assert result.exit_code == 0
