from pathlib import Path

import pytest

from nos.common.git import cached_repo


def test_common_cached_repo():
    repo_url = "https://github.com/google/styleguide.git"
    tag = "gh-pages"

    subdir = cached_repo(repo_url, force=True)
    assert subdir is not None
    assert Path(subdir).exists()

    subdir = cached_repo(repo_url, subdirectory="go/", tag=tag, force=True)
    assert subdir is not None
    assert Path(subdir).exists()

    subdir = cached_repo(
        f"https://github.com/google/styleguide/archive/refs/heads/{tag}.zip",
        repo_name="styleguide",
        subdirectory="go/",
        force=True,
    )
    assert subdir is not None
    assert Path(subdir).exists()

    with pytest.raises(ValueError):
        subdir = cached_repo(repo_url.replace(".git", ""), subdirectory="go/", tag="fake-tag")

    with pytest.raises(ValueError):
        subdir = cached_repo(repo_url, branch="2.x", tag="2.x")
