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
    import os

    if os.path.exists("./test-catalog/"):
        shutil.rmtree("./test-catalog/")

    result = runner.invoke(app_cli, ["profile", "model", "-m", "openai/clip", "--catalog-path", "./test-catalog/"])
    assert result.exit_code == 0

    assert os.path.exists("./test-catalog/")
    # load the json manually and check the field is not 'nan':
    import json

    with open(os.path.join("./test-catalog/", os.listdir("./test-catalog/")[0])) as f:
        catalog = json.load(f)
    assert catalog[0]["profiling_data"]["forward_warmup::execution"]["gpu_utilization"] != "nan"
    shutil.rmtree("./test-catalog/")


@pytest.mark.benchmark(group=PyTestGroup.MODEL_PROFILE)
def test_cli_profile_all():
    result = runner.invoke(app_cli, ["profile", "all"])
    assert result.exit_code == 0
