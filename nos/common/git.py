import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import git

from nos.constants import NOS_CACHE_DIR
from nos.logging import logger


def cached_repo(
    repo_url: str,
    repo_name: str = None,
    subdirectory: str = None,
    branch: str = None,
    tag: str = None,
    depth: int = None,
    force: bool = False,
) -> str:
    """Clone a repository and cache it locally.

    Args:
        repo_url (str): The URL of the repository to clone.
        repo_name (str): The name of the repository. If None, the name is inferred from the URL.
        subdirectory (str): The subdirectory within the repository to cache.
            If None, the entire repository is cached.
        branch (str): The branch to checkout.
        tag (str): The tag to checkout.
        depth (int): The depth to clone the repository.
        force (bool): If True, the repository will be cloned even if it already exists.

    Returns:
        str: The path to the cached directory.
    """
    if not (repo_url.endswith(".git") or repo_url.startswith("https://")):
        raise ValueError("Invalid repository URL, needs to end with `.git` or start with `https://`.")

    if branch and tag:
        raise ValueError("Cannot specify both branch and tag, only specify one.")

    # Determine the cached repo location
    if repo_url.endswith(".git"):
        repo_name = Path(repo_url).stem
    else:
        if not repo_name:
            raise ValueError("Must specify `repo_name` for HTTPS repositories.")
        repo_name = repo_name

    cached_repos_dir = NOS_CACHE_DIR / "repos"
    cached_repos_dir.mkdir(parents=True, exist_ok=True)
    cached_dir = cached_repos_dir / repo_name

    cached_subdir = None
    if subdirectory:
        cached_subdir = cached_dir / subdirectory

    if force:
        try:
            shutil.rmtree(str(cached_dir))
        except FileNotFoundError:
            pass

    if cached_dir.exists():
        return str(cached_subdir or cached_dir)

    # Create a temporary directory to clone the repository
    temp_dir = Path(tempfile.mkdtemp())

    # Create a lock file to prevent multiple processes from cloning the same repository
    lock_file = cached_repos_dir / f"{repo_name}.lock"
    if lock_file.exists():
        raise ValueError(f"Repository `{repo_url}` is already being cloned.")
    lock_file.touch()

    # Clone the repository, and copy the contents to the cache directory.
    # Cleanup the temporary directory and remove the lock file once done.
    try:
        if repo_url.startswith("https://") and repo_url.endswith(".zip"):
            if branch or tag:
                raise ValueError("Cannot specify branch or tag for HTTPS repositories.")

            # Download the repository using HTTPS
            zip_filename = temp_dir / "repo.zip"
            logger.debug(f"Downloading repository [repo={repo_url}, dir={cached_dir}]")
            urllib.request.urlretrieve(repo_url, zip_filename)
            logger.debug(f"Downloaded repository [repo={repo_url}, dir={cached_dir}]")

            # Unzip the repository
            with zipfile.ZipFile(str(zip_filename), "r") as zip_ref:
                zip_ref.extractall(temp_dir)
            parent = zip_ref.namelist()[0]
            logger.debug(f"Unzipped repository [repo={repo_url}, dir={str(temp_dir / parent)}]")
            shutil.move(str(temp_dir / parent), str(cached_dir))

            # Remove the zip file
            zip_filename.unlink()

        elif repo_url.endswith(".git"):
            logger.debug(f"Cloning repository [repo={repo_url}, dir={cached_dir}]")
            repo = git.Repo.clone_from(repo_url, temp_dir, depth=depth)
            logger.debug(f"Cloned repository [repo={repo_url}, dir={cached_dir}]")

            # Checkout the specified branch or tag
            if branch:
                repo.git.checkout(branch)
                logger.debug(f"Checked out branch [repo={repo_url}, branch={branch}]")
            elif tag:
                repo.git.checkout(tag)
                logger.debug(f"Checked out tag [repo={repo_url}, tag={tag}]")

        else:
            raise ValueError(f"Invalid repository URL: {repo_url}")

        # Move the contents to the cache directory
        logger.debug(f"Moving repository [repo={repo_url}, dir={cached_dir}]")
        shutil.move(str(temp_dir), str(cached_dir))

        # Navigate to the subdirectory within the cloned repository
        if subdirectory:
            cached_subdir = cached_dir / subdirectory
            if not cached_subdir.exists():
                raise ValueError(f"Subdirectory `{subdirectory}`does not exist in the repository.")
    finally:
        # Delete the temporary directory
        try:
            shutil.rmtree(str(temp_dir))
        except FileNotFoundError:
            pass

        # The lock file will be removed Remove the lock file
        lock_file.unlink()

    assert cached_dir.exists(), f"Repository `{repo_url}` was not cached."
    if subdirectory:
        assert cached_subdir.exists(), f"Subdirectory `{subdirectory}` was not cached."
    return str(cached_subdir or cached_dir)
