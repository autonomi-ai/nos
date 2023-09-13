import tempfile
from pathlib import Path

import pytest

from nos.common.git import cached_repo


def test_common_cached_repo():
    repo_url = "https://github.com/huggingface/diffusers.git"
    tag = "v0.20.1"

    subdir = cached_repo(repo_url, force=True)
    assert subdir is not None
    assert Path(subdir).exists()

    subdir = cached_repo(repo_url, subdirectory="examples/dreambooth", tag=tag, force=True)
    assert subdir is not None
    assert Path(subdir).exists()

    subdir = cached_repo(
        f"https://github.com/huggingface/diffusers/archive/refs/tags/{tag}.zip",
        repo_name="diffusers",
        subdirectory="examples/dreambooth",
        force=True,
    )
    assert subdir is not None
    assert Path(subdir).exists()

    with tempfile.TemporaryDirectory() as tmpdir:
        subdir = cached_repo(repo_url, subdirectory="examples/dreambooth", tag=tag, force=True, output_dir=tmpdir)
        assert subdir is not None
        assert Path(subdir).exists()

    with pytest.raises(ValueError):
        subdir = cached_repo(repo_url.replace(".git", ""), subdirectory="examples/dreambooth", tag="v0.20.1")

    with pytest.raises(ValueError):
        subdir = cached_repo(repo_url, branch="2.x", tag="2.x")
