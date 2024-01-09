# Copyright 2023 Autonomi AI, Inc. All rights reserved.
AGIPACK_ARGS :=

agi-build-cpu:  # equivalent to agi-build-py38-cpu (target=base-cpu, cpu, test-cpu)
	agi-pack build ${AGIPACK_ARGS} \
		--target cpu \
		-c docker/agibuild.cpu.yaml \
		-o docker/Dockerfile.cpu \
		-p 3.8.15 \
		-b debian:buster-slim \
		-t '${DOCKER_IMAGE_NAME}:${NOS_VERSION_TAG}-cpu'
	docker tag \
		${DOCKER_IMAGE_NAME}:${NOS_VERSION_TAG}-cpu \
		${DOCKER_IMAGE_NAME}:latest-cpu

agi-build-gpu:  # equivalent to agi-build-py38-cu118 (target=base-gpu, gpu, test-gpu)
	agi-pack build ${AGIPACK_ARGS} \
		--target gpu \
		-c docker/agibuild.gpu.yaml \
		-o docker/Dockerfile.gpu \
		-p 3.8.15 \
		-b nvidia/cuda:11.8.0-base-ubuntu22.04 \
		-t '${DOCKER_IMAGE_NAME}:${NOS_VERSION_TAG}-gpu'
	docker tag \
		${DOCKER_IMAGE_NAME}:${NOS_VERSION_TAG}-gpu \
		${DOCKER_IMAGE_NAME}:latest-gpu

agi-build-py38-cu118:
	agi-pack build ${AGIPACK_ARGS} \
		--target gpu \
		-c docker/agibuild.gpu.yaml \
		-o docker/Dockerfile.py38-cu118 \
		-p 3.8.15 \
		-b nvidia/cuda:11.8.0-base-ubuntu22.04 \
		-t '${DOCKER_IMAGE_NAME}:${NOS_VERSION_TAG}-py38-cu118'
	docker tag \
		${DOCKER_IMAGE_NAME}:${NOS_VERSION_TAG}-py38-cu118 \
		${DOCKER_IMAGE_NAME}:latest-py38-cu118

agi-build-py39-cu118:
	agi-pack build ${AGIPACK_ARGS} \
		--target gpu \
		-c docker/agibuild.gpu.yaml \
		-o docker/Dockerfile.py39-cu118 \
		-p 3.9.13 \
		-b nvidia/cuda:11.8.0-base-ubuntu22.04 \
		-t '${DOCKER_IMAGE_NAME}:${NOS_VERSION_TAG}-py39-cu118'
	docker tag \
		${DOCKER_IMAGE_NAME}:${NOS_VERSION_TAG}-py39-cu118 \
		${DOCKER_IMAGE_NAME}:latest-py39-cu118

agi-build-py310-cu118:
	agi-pack build ${AGIPACK_ARGS} \
		--target gpu \
		-c docker/agibuild.gpu.yaml \
		-o docker/Dockerfile.py310-cu118 \
		-p 3.10.11 \
		-b nvidia/cuda:11.8.0-base-ubuntu22.04 \
		-t '${DOCKER_IMAGE_NAME}:${NOS_VERSION_TAG}-py310-cu118'
	docker tag \
		${DOCKER_IMAGE_NAME}:${NOS_VERSION_TAG}-py310-cu118 \
		${DOCKER_IMAGE_NAME}:latest-py310-cu118

agi-build-py311-cu118:
	agi-pack build ${AGIPACK_ARGS} \
		--target gpu \
		-c docker/agibuild.gpu.yaml \
		-o docker/Dockerfile.py310-cu118 \
		-p 3.11.4 \
		-b nvidia/cuda:11.8.0-base-ubuntu22.04 \
		-t '${DOCKER_IMAGE_NAME}:${NOS_VERSION_TAG}-{target}-py311-cu118'
	docker tag \
		${DOCKER_IMAGE_NAME}:${NOS_VERSION_TAG}-py311-cu118 \
		${DOCKER_IMAGE_NAME}:latest-py311-cu118

agi-build-py38-cu121:
	agi-pack build ${AGIPACK_ARGS} \
		--target gpu \
		-c docker/agibuild.cu121.yaml \
		-o docker/Dockerfile.py38-cu121 \
		-p 3.8.15 \
		-b nvidia/cuda:12.1.1-base-ubuntu22.04 \
		-t '${DOCKER_IMAGE_NAME}:${NOS_VERSION_TAG}-py38-cu121'
	docker tag \
		${DOCKER_IMAGE_NAME}:${NOS_VERSION_TAG}-py38-cu121 \
		${DOCKER_IMAGE_NAME}:latest-py38-cu121

agi-build-py310-cu121:
	agi-pack build ${AGIPACK_ARGS} \
		--target gpu \
		-c docker/agibuild.cu121.yaml \
		-o docker/Dockerfile.py310-cu121 \
		-p 3.10.11 \
		-b nvidia/cuda:12.1.1-base-ubuntu22.04 \
		-t '${DOCKER_IMAGE_NAME}:${NOS_VERSION_TAG}-py310-cu121'
	docker tag \
		${DOCKER_IMAGE_NAME}:${NOS_VERSION_TAG}-py310-cu121 \
		${DOCKER_IMAGE_NAME}:latest-py310-cu121

agi-build-cu118: \
	agi-build-py38-cu118 \
	agi-build-py39-cu118 \
	agi-build-py310-cu118 \
	agi-build-py311-cu118

agi-build-cu121: \
	agi-build-py38-cu121 \
	agi-build-py310-cu121

agi-push-cu118:
	docker push ${DOCKER_IMAGE_NAME}:${NOS_VERSION_TAG}-py38-cu118
	docker push ${DOCKER_IMAGE_NAME}:latest-py38-cu118
	docker push ${DOCKER_IMAGE_NAME}:${NOS_VERSION_TAG}-py39-cu118
	docker push ${DOCKER_IMAGE_NAME}:latest-py39-cu118
	docker push ${DOCKER_IMAGE_NAME}:${NOS_VERSION_TAG}-py310-cu118
	docker push ${DOCKER_IMAGE_NAME}:latest-py310-cu118
	docker push ${DOCKER_IMAGE_NAME}:${NOS_VERSION_TAG}-py311-cu118
	docker push ${DOCKER_IMAGE_NAME}:latest-py311-cu118

agi-push-cu121:
	docker push ${DOCKER_IMAGE_NAME}:${NOS_VERSION_TAG}-py38-cu121
	docker push ${DOCKER_IMAGE_NAME}:latest-py38-cu121
	docker push ${DOCKER_IMAGE_NAME}:${NOS_VERSION_TAG}-py310-cu121
	docker push ${DOCKER_IMAGE_NAME}:latest-py310-cu121
