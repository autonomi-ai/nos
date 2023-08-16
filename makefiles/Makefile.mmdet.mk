.docker-build-mmdet:
	docker build -f docker/Dockerfile.mmdet \
		--target test \
		-t ${DOCKER_IMAGE_NAME}:latest-mmdet-${TARGET} \
		-t ${DOCKER_IMAGE_NAME}:${NOS_VERSION_TAG}-mmdet-${TARGET} \
		--build-arg BASE_IMAGE=${BASE_IMAGE} \
		.

.docker-run-mmdet:
	docker run -it ${DOCKER_ARGS} --rm \
		${DOCKER_IMAGE_NAME}:${NOS_VERSION_TAG}-mmdet-${TARGET} \
		${DOCKER_CMD}

docker-build-mmdet-dev:
	make docker-build-gpu DOCKER_TARGET=base
	make .docker-build-mmdet \
	TARGET=dev BASE_IMAGE=autonomi/nos:latest-gpu

docker-run-mmdet-grpc-server: docker-build-mmdet-dev
	make .docker-run-mmdet TARGET=dev \
	DOCKER_ARGS="--gpus all -v $(shell pwd):/nos -v ${HOME}/.nosd:/app/.nos -p 50051:50051 -p 8265:8265" \
	DOCKER_CMD=""

docker-run-mmdet-interactive: docker-build-mmdet-dev
	make .docker-run-mmdet TARGET=dev \
	DOCKER_ARGS="--gpus all -v $(shell pwd):/nos -v ${HOME}/.nosd:/app/.nos -p 50051:50051 -p 8265:8265" \
	DOCKER_CMD="bash"

docker-test-mmdet:
	docker compose -f docker-compose.extras.yml run --rm --build test-mmdet-dev
