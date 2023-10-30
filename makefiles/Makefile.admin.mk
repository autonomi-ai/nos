# Copyright 2023 Autonomi AI, Inc. All rights reserved.

NOS_VERSION := $(shell python -c 'from nos.version import __version__; print(__version__.replace("-", "."))')
PYPI_USERNAME :=
PYPI_PASSWORD :=

WHL_GREP_PATTERN := .*\$(NOS_VERSION).*\.whl

create-pypi-release-test:
	@echo "looking for nos whl file..."
	@for file in dist/*; do \
		echo "examining file: $$file"; \
		if [ -f "$$file" ] && echo "$$file" | grep -qE "$(WHL_GREP_PATTERN)"; then \
			echo "Uploading: $$file"; \
			twine upload --repository testpypi "$$file" --username $(PYPI_USERNAME) --password $(PYPI_PASSWORD); \
		fi; \
	done
	@echo "Upload completed"


create-pypi-release:
	@echo "looking for nos whl file..."
	@for file in dist/*; do \
		echo "examining file: $$file"; \
		if [ -f "$$file" ] && echo "$$file" | grep -qE "$(WHL_GREP_PATTERN)"; then \
			echo "Uploading: $$file"; \
			twine upload "$$file" --username $(PYPI_USERNAME) --password $(PYPI_PASSWORD); \
		fi; \
	done
	@echo "Upload completed"

create-tag:
	git tag -a ${NOS_VERSION} -m "Release ${NOS_VERSION}"
	git push origin ${NOS_VERSION}

docker-login:  ## Login to Docker Hub
	docker login
