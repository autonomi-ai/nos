# Copyright 2022- Autonomi AI, Inc. All rights reserved.
#
NOS_VERSION := $(shell python -c 'from nos.version import __version__; print(__version__)')

create-pypi-release-test:  ## package and upload a release
	twine upload --repository testpypi dist/autonomi_nos-${NOS_VERSION}-py3-none-any.whl --

create-pypi-release:  ## package, git tag/release and upload a release to PyPI
	@echo -n "Are you sure you want to create a PyPI release? [y/N] " && read ans && [ $${ans:-N} = y ]
	echo "Uploading dist/autonomi_nos-${NOS_VERSION}-py3-none-any.whl"
	twine upload dist/autonomi_nos-${NOS_VERSION}-py3-none-any.whl
	echo "Successfully created release ${NOS_VERSION}."
