from pathlib import Path

import pytest

from nos.common.git import cached_repo


def test_common_cached_repo():
    repo_url = "https://github.com/open-mmlab/mmdetection.git"

    subdir = cached_repo(repo_url)
    assert subdir is not None
    assert Path(subdir).exists()

    subdir = cached_repo(repo_url, subdirectory="configs", branch="2.x")
    assert subdir is not None
    assert Path(subdir).exists()

    with pytest.raises(ValueError):
        subdir = cached_repo(repo_url.replace(".git", ""), subdirectory="configs", branch="2.x")

    with pytest.raises(ValueError):
        subdir = cached_repo(repo_url, branch="2.x", tag="2.x")
