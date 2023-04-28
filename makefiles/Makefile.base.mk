NOS_VERSION := $(shell sed -n 's/^__version__ = "\([0-9.]\+\)"/\1/p' nos/version.py)
NOS_VERSION_TAG := v${NOS_VERSION}
PYTHON_VERSION := $(shell python -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')
DOCKER_BASE_IMAGE_NAME := autonomi-ai/nos
DOCKER_ARGS :=
DOCKER_CMD :=


.docker-build-base:
	docker build -f docker/Dockerfile.base \
		-t ${DOCKER_BASE_IMAGE_NAME}:latest-${TARGET}-base \
		-t ${DOCKER_BASE_IMAGE_NAME}:${NOS_VERSION_TAG}-${TARGET}-base \
		--build-arg BASE_IMAGE=${BASE_IMAGE} \
		.

.docker-run-base:
	docker run -it ${DOCKER_ARGS} --rm \
		${DOCKER_BASE_IMAGE_NAME}:${NOS_VERSION_TAG}-${TARGET}-base \
		${DOCKER_CMD}

docker-build-base-py3-cpu:
	make .docker-build-base TARGET=cpu \
	PYTHON_VERSION=${PYTHON_VERSION} \
	BASE_IMAGE=python:${PYTHON_VERSION}-slim

docker-build-base-py3-gpu:
	make .docker-build-base \
	TARGET=gpu \
	PYTHON_VERSION=${PYTHON_VERSION} \
	BASE_IMAGE=nvidia/cuda:11.8.0-runtime-ubuntu22.04

docker-run-base-all: \
	docker-run-base-py3-cpu docker-run-base-py3-gpu

docker-compose-upd-py3-gpu: docker-build-base-py3-gpu
	docker compose -f docker-compose.yml up
