NOS_VERSION := $(shell sed -n 's/^__version__ = "\([0-9.]\+\)"/\1/p' nos/version.py)
NOS_VERSION_TAG := v${NOS_VERSION}

DOCKER_BASE_IMAGE_NAME := nos
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

# Python 3.8.10
docker-build-base-py38-cpu:
	make .docker-build-base TARGET=cpu BASE_IMAGE=python:3.8.10-slim

# Python 3.8.10
# tensorrt:23.02-py3
# NVIDIA Release 23.02 (build 52693241)
# NVIDIA TensorRT Version 8.5.3
# See support matrix: https://docs.nvidia.com/deeplearning/tensorrt/container-release-notes/index.html#rel-23-03
docker-build-base-py38-trt:
	make .docker-build-base TARGET=trt BASE_IMAGE=nvcr.io/nvidia/tensorrt:23.02-py3

# Python 3.8.10
# pytorch: 1.14.0a0+44dac51
# tensorrt:23.02-py3
# NVIDIA Release 23.02 (build 52693241)
# NVIDIA TensorRT Version 8.5.3
# See: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch
docker-build-base-py38-nvidia-pytorch:
	make .docker-build-base TARGET=nvidia-pytorch BASE_IMAGE=nvcr.io/nvidia/pytorch:23.02-py3

docker-build-base-all: \
	docker-build-base-py38-cpu docker-build-base-py38-trt docker-build-base-py38-nvidia-pytorch

docker-run-base-py38-cpu:
	make .docker-run-base TARGET=cpu DOCKER_CMD="env | grep NOS"

docker-run-base-py38-trt:
	make .docker-run-base TARGET=trt \
	DOCKER_ARGS="--gpus all" DOCKER_CMD="nvidia-smi && env | grep NOS"

docker-run-base-py38-nvidia-pytorch:
	make .docker-run-base TARGET=nvidia-pytorch \
	DOCKER_ARGS="--gpus all" DOCKER_CMD="nvidia-smi && env | grep NOS"

docker-run-base-all: \
	docker-run-base-py38-cpu docker-run-base-py38-trt docker-run-base-py38-nvidia-pytorch
