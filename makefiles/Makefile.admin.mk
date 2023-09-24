# Copyright 2022- Autonomi AI, Inc. All rights reserved.
#
NOS_VERSION := $(shell python -c 'from nos.version import __version__; print(__version__.replace("-", "."))')
PYPI_USERNAME :=
PYPI_PASSWORD :=

# twine upload .whl files in the directory that contain the string NOS_VERSION:
create-pypi-release-loose-test:
	twine upload --repository testpypi dist/*${NOS_VERSION}*.whl --

WHL_GREP_PATTERN := .*\$(NOS_VERSION).*\.whl

create-pypi-release-loose-test:
	@echo "looking for nos whl file..."
	@for file in dist/*; do \
		echo "examining file: $$file"; \
		if [ -f "$$file" ] && echo "$$file" | grep -qE "$(WHL_GREP_PATTERN)"; then \
			echo "Uploading: $$file"; \
			twine upload --repository testpypi "$$file" --username $(PYPI_USERNAME) --password $(PYPI_PASSWORD); \
		fi; \
	done
	@echo "Upload completed"

create-pypi-release:  ## package, git tag/release and upload a release to PyP I
	@echo -n "Are you sure you want to create a PyPI release? [y/N] " && read ans && [ $${ans:-N} = y ]
	echo "Uploading dist/torch_nos-${NOS_VERSION}-py3-none-any.whl"
	twine upload dist/torch_nos-${NOS_VERSION}-py3-none-any.whl
	echo "Successfully created release ${NOS_VERSION}."

create-tag:
	git tag -a ${NOS_VERSION} -m "Release ${NOS_VERSION}"
	git push origin ${NOS_VERSION}

docker-login:  ## Login to Docker Hub
	docker login
