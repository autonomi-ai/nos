export DOCKER_BUILDKIT ?= 1
export COMPOSE_DOCKER_CLI_BUILD ?= 1

NOS_VERSION := $(shell sed -n 's/^__version__ = "\([0-9.]\+\)"/\1/p' nos/version.py)
NOS_VERSION_TAG := ${NOS_VERSION}
DOCKER_IMAGE_NAME := autonomi/nos
DOCKER_TARGET := base
DOCKER_ARGS :=
DOCKER_CMD :=

help-base:
	@echo "nos 🔥: Nitrous Oxide System (NOS) for Computer Vision"
	@echo ""
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets:"
	@echo "  docker-build-cpu    Build CPU docker image"
	@echo "  docker-build-gpu    Build GPU docker image"
	@echo "  docker-build-all    Build CPU and GPU docker images"
	@echo "  docker-test-cpu     Test CPU docker image"
	@echo "  docker-test-gpu     Test GPU docker image"
	@echo "  docker-test-all     Test CPU and GPU docker images"
	@echo "  docker-compose-upd-gpu    Run GPU docker image"
	@echo "  docker-compose-upd-cpu    Run CPU docker image"
	@echo "  docker-push-cpu     Push CPU docker image"
	@echo "  docker-push-gpu     Push GPU docker image"
	@echo "  docker-push-all     Push CPU and GPU docker images"
	@echo ""

.docker-build:
	@echo "🛠️ Building docker image"
	@echo "BASE_IMAGE: ${BASE_IMAGE}"
	@echo "TARGET: ${TARGET}"
	@echo "DOCKER_TARGET: ${DOCKER_TARGET}"
	@echo "IMAGE: ${DOCKER_IMAGE_NAME}:${NOS_VERSION_TAG}-${TARGET}"
	@echo ""
	docker build -f docker/Dockerfile \
		--target ${DOCKER_TARGET} \
		-t ${DOCKER_IMAGE_NAME}:latest-${TARGET} \
		-t ${DOCKER_IMAGE_NAME}:${NOS_VERSION_TAG}-${TARGET} \
		--build-arg BASE_IMAGE=${BASE_IMAGE} \
		--build-arg TARGET=${TARGET} \
		.

.docker-run:
	docker run -it ${DOCKER_ARGS} --rm \
		${DOCKER_IMAGE_NAME}:${NOS_VERSION_TAG}-${TARGET} \
		${DOCKER_CMD}

.docker-push-base:
	docker push ${DOCKER_IMAGE_NAME}:latest-${TARGET}
	docker push ${DOCKER_IMAGE_NAME}:${NOS_VERSION_TAG}-${TARGET}

docker-build-cpu:
	make .docker-build \
	TARGET=cpu \
	DOCKER_TARGET=${DOCKER_TARGET} \
	BASE_IMAGE=python:3.8.10-slim

docker-build-gpu:
	make .docker-build \
	TARGET=gpu \
	DOCKER_TARGET=${DOCKER_TARGET} \
	BASE_IMAGE=nvidia/cuda:11.8.0-base-ubuntu22.04

docker-build-all: \
	docker-build-cpu docker-build-gpu

docker-test-cpu:
	docker compose -f docker-compose.test.yml run --rm --build test-cpu

docker-test-gpu:
	docker compose -f docker-compose.test.yml run --rm --build test-gpu

docker-test-all: \
	docker-test-cpu docker-test-gpu

docker-push-cpu:
	make .docker-push-base \
	TARGET=cpu

docker-push-gpu:
	make .docker-push-base \
	TARGET=gpu

docker-push-all: \
	docker-push-cpu docker-push-gpu

docker-compose-upd-cpu: docker-build-cpu
	docker compose -f docker-compose.cpu.yml up

docker-compose-upd-gpu: docker-build-gpu
	docker compose -f docker-compose.gpu.yml up
