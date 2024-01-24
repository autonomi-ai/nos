import shutil

import pytest
from typer.testing import CliRunner

from nos.cli.cli import app_cli
from nos.test.utils import PyTestGroup, skip_if_no_torch_cuda


runner = CliRunner()


def test_cli_profile_list():
    result = runner.invoke(app_cli, ["profile", "list"])
    assert result.exit_code == 0


@skip_if_no_torch_cuda
def test_cli_profile_model():
    result = runner.invoke(app_cli, ["profile", "model", "-m", "openai/clip"])
    assert result.exit_code == 0


@skip_if_no_torch_cuda
def test_catalog_path():
    result = runner.invoke(app_cli, ["profile", "model", "-m", "openai/clip", "--catalog-path", "./test-catalog/"])
    assert result.exit_code == 0
    import os

    assert len(os.listdir("./test-catalog/")) > 0
    shutil.rmtree("./test-catalog/")


@skip_if_no_torch_cuda
def test_gpu_util():
    result = runner.invoke(app_cli, ["profile", "model", "-m", "openai/clip", "--catalog-path", "./test-catalog/"])
    assert result.exit_code == 0
    import os

    assert os.path.exists("./test-catalog/")
    # load the json manually and check the field is not 'nan':
    import json

    # open the first file in the tests/ dir:
    # open the absolute path:
    with open(os.path.join("./test-catalog/", os.listdir("./test-catalog/")[0])) as f:
        catalog = json.load(f)
    assert catalog["openai/clip"]["gpu_util"] != "nan"
    shutil.rmtree("./test-catalog/")


@pytest.mark.benchmark(group=PyTestGroup.MODEL_PROFILE)
def test_cli_profile_all():
    result = runner.invoke(app_cli, ["profile", "all"])
    assert result.exit_code == 0
