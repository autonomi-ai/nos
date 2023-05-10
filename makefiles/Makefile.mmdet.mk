export DOCKER_BUILDKIT ?= 1
export COMPOSE_DOCKER_CLI_BUILD ?= 1

NOS_VERSION := $(shell sed -n 's/^__version__ = "\([0-9.]\+\)"/\1/p' nos/version.py)
NOS_VERSION_TAG := v${NOS_VERSION}
PYTHON_VERSION := $(shell python -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')
DOCKER_IMAGE_NAME := autonomi-ai/nos
DOCKER_ARGS :=
DOCKER_CMD :=


.docker-build-mmdet:
	docker build -f docker/Dockerfile.mmdet \
		-t ${DOCKER_IMAGE_NAME}:latest-mmdet-${TARGET} \
		-t ${DOCKER_IMAGE_NAME}:${NOS_VERSION_TAG}-mmdet-${TARGET} \
		--build-arg BASE_IMAGE=${BASE_IMAGE} \
		--build-arg TARGET=${TARGET} \
		.

.docker-run-mmdet:
	docker run -it ${DOCKER_ARGS} --rm \
		${DOCKER_IMAGE_NAME}:${NOS_VERSION_TAG}-mmdet-${TARGET} \
		${DOCKER_CMD}

docker-build-mmdet-gpu:
	make .docker-build-mmdet \
	TARGET=gpu

docker-run-mmdet-grpc-server: docker-build-mmdet-gpu
	make .docker-run-mmdet TARGET=gpu \
	DOCKER_ARGS="--gpus all -v $(shell pwd):/nos -v ${HOME}/.nos_docker:/app/.nos -p 50051:50051 -p 8265:8265" \
	DOCKER_CMD="nos-grpc-server"
