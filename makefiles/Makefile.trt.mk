# TARGET: (trt-dev, trt-runtime)
.docker-build-trt:  # Build the trt docker image
	docker build -f docker/Dockerfile.trt \
		--target test-${TARGET} \
		-t ${DOCKER_IMAGE_NAME}:latest-${TARGET} \
		-t ${DOCKER_IMAGE_NAME}:${NOS_VERSION_TAG}-${TARGET} \
		--build-arg BASE_IMAGE=${BASE_IMAGE} \
		.

# Note: Currently docker compose does not respect pull policy, so it downloads
# the image every time. For this reason, we manually build the image
# and then run docker compose.
# .docker-build-trt:
# 	docker compose -f docker-compose.extras.yml build trt-dev

# TARGET: (trt-dev, trt-runtime)
.docker-run-trt:
	docker run -it ${DOCKER_ARGS} --rm \
		${DOCKER_IMAGE_NAME}:${NOS_VERSION_TAG}-${TARGET} \
		${DOCKER_CMD}

docker-build-trt-dev:
	make docker-build-gpu DOCKER_TARGET=base
	make .docker-build-trt \
	TARGET=trt-dev BASE_IMAGE=autonomi/nos:latest-gpu

docker-build-trt-runtime:
	make docker-build-gpu DOCKER_TARGET=base
	make .docker-build-trt \
	TARGET=trt-runtime BASE_IMAGE=autonomi/nos:latest-gpu

docker-run-trt-notebook:
	docker run -it --gpus all \
		-p 8888:8888 \
		-e NOS_HOME=/app/.nos \
		-e NOS_LOG_LEVEL=DEBUG \
		-v ~/.nosd:/app/.nos \
		-v ${PWD}:/app/nos/ \
		-v /tmp/trt/.cache:/root/.cache \
		autonomi/nos:latest-trt-dev bash -c "cd /app/nos/examples/notebook && jupyter notebook --ip=0.0.0.0 --allow-root"

docker-test-trt:
	docker compose -f docker-compose.extras.yml run --rm --build trt-dev
