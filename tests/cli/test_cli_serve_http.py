import pytest
from typer.testing import CliRunner

from nos.cli.cli import app_cli


runner = CliRunner()


@pytest.mark.skip(reason="Not yet implemented.")
def test_cli_serve_list():
    result = runner.invoke(app_cli, ["serve", "list"])
    assert result.exit_code == 0
