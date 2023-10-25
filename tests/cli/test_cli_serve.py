import os
from pathlib import Path
from typing import Union

import pytest
from typer.testing import CliRunner


pytestmark = pytest.mark.cli
runner = CliRunner()


def path_ctx(path: Union[Path, str]):
    prev_cwd = Path.cwd()
    os.chdir(str(path))
    yield
    os.chdir(str(prev_cwd))


def test_cli_serve_help():
    from nos.cli.cli import app_cli

    result = runner.invoke(app_cli, ["serve", "--help"])
    assert result.exit_code == 0


@pytest.mark.skip
def test_cli_serve_up():

    from nos.cli.cli import app_cli
    from nos.tests.utils import NOS_TEST_DATA_DIR

    # Move to the test data directory
    path = NOS_TEST_DATA_DIR / "hub/custom_model/config.yaml"

    with path_ctx(path.parent):
        result = runner.invoke(app_cli, ["serve", "-c", "config.yaml", "up"])
        assert result.exit_code == 0
