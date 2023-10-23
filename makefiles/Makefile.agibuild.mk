
agi-build-cpu:
	agi-pack build \
		-c docker/agibuild.cpu.yaml \
		-o docker/Dockerfile.cpu \
		-t '${DOCKER_IMAGE_NAME}:${NOS_VERSION_TAG}-{target}'

agi-build-gpu:
	agi-pack build \
		-c docker/agibuild.gpu.yaml \
		-o docker/Dockerfile.gpu \
		-t '${DOCKER_IMAGE_NAME}:${NOS_VERSION_TAG}-{target}'

agi-build-py310-cu118:
	agi-pack build \
		-c docker/agibuild.gpu.yaml \
		-o docker/Dockerfile.py310-cu118 \
		-p 3.10 \
		-b nvidia/cuda:11.8.0-base-ubuntu22.04 \
		-t '${DOCKER_IMAGE_NAME}:${NOS_VERSION_TAG}-{target}-py310-cu118'
