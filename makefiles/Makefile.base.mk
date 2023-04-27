export DOCKER_BUILDKIT ?= 1
export COMPOSE_DOCKER_CLI_BUILD ?= 1

NOS_VERSION := $(shell sed -n 's/^__version__ = "\([0-9.]\+\)"/\1/p' nos/version.py)
NOS_VERSION_TAG := v${NOS_VERSION}
PYTHON_VERSION := $(shell python -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')
DOCKER_IMAGE_NAME := autonomi-ai/nos
DOCKER_ARGS :=
DOCKER_CMD :=


.docker-build:
	docker build -f docker/Dockerfile \
		-t ${DOCKER_IMAGE_NAME}:latest-${TARGET} \
		-t ${DOCKER_IMAGE_NAME}:${NOS_VERSION_TAG}-${TARGET} \
		--build-arg BASE_IMAGE=${BASE_IMAGE} \
		--build-arg TARGET=${TARGET} \
		.

.docker-run:
	docker run -it ${DOCKER_ARGS} --rm \
		${DOCKER_IMAGE_NAME}:${NOS_VERSION_TAG}-${TARGET} \
		${DOCKER_CMD}

docker-build-cpu:
	make .docker-build TARGET=cpu \
	BASE_IMAGE=python:3.8.10-slim

docker-build-gpu:
	make .docker-build \
	TARGET=gpu \
	BASE_IMAGE=nvidia/cuda:11.8.0-base-ubuntu22.04

docker-build-all: \
	docker-build-cpu docker-build-gpu

docker-compose-upd-gpu: docker-build-gpu
	docker compose -f docker-compose.gpu.yml up

docker-test-cpu: docker-build-cpu
	make .docker-run TARGET=cpu \
	DOCKER_ARGS="-v $(shell pwd):/nos" \
	DOCKER_CMD="make test-cpu"

docker-test-gpu: docker-build-gpu
	make .docker-run TARGET=gpu \
	DOCKER_ARGS="-v $(shell pwd):/nos --gpus all" \
	DOCKER_CMD="make test"
