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
	DOCKER_ARGS="--gpus all -v $(shell pwd):/nos -v ${HOME}/.nosd:/app/.nos -p 50051:50051 -p 8265:8265" \
	DOCKER_CMD="nos-grpc-server"
