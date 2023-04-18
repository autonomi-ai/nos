NOS_VERSION := $(shell sed -n 's/^__version__ = "\([0-9.]\+\)"/\1/p' nos/_version.py)
NOS_VERSION_TAG := v${NOS_VERSION}

TORCHSERVE_VERSION := 0.7.1

DOCKER_BASE_IMAGE_NAME := nos
DOCKER_ARGS :=
DOCKER_CMD :=

# FastRCNN serving with torch-serve
# 1. Download fasterrcnn_resnet50_fpn_coco-258fb6c6.pth from https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth
# 2. `docker-run-torch-serve-archive`: Builds the fast-rcnn.mar model archive
# 3. `docker-run-torch-serve`: Serves the model on 8080
# 4.
docker-run-torch-serve-archive:
	@echo "mkdir -p ~/.nos/models"
	@echo "wget https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth ~/.nos/models/"
	MODEL_NAME=fastrcnn && \
	docker run -it --rm --gpus all \
		-v ${PWD}/docker/open-mmlab/mmdet/serve/archive:/opt/nos/archive \
		-v ${HOME}/.nos/models:/models \
		-v ${PWD}/nos:/opt/nos \
		${DOCKER_BASE_IMAGE_NAME}:${NOS_VERSION_TAG}-mmdet-server \
		torch-model-archiver \
		--model-name fastrcnn \
		--version 1.0 \
		--handler object_detector \
		--serialized-file /models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth \
		--model-file /opt/nos/externals/models/fast-rcnn/model.py \
		--extra-files /opt/nos/externals/models/fast-rcnn/index_to_name.json \
		--export-path /models/serve/ --force

# Serves the fastrcnn.mar model archive on 8080 using pytorch/torchserve:0.7.1-gpu
# Currently this runs in daemon mode
docker-run-torch-serve:
	MODEL_NAME=fast-rcnn \
	docker run -d --rm --gpus all \
		-p 8080:8080 -p 8081:8081 -p 8082:8082 \
		-v ${PWD}/docker/open-mmlab/mmdet/serve/archive:/opt/nos/archive \
		-v ${HOME}/.nos/models:/models \
		-v ${PWD}/nos:/opt/nos \
		pytorch/torchserve:${TORCHSERVE_VERSION}-gpu \
		torchserve --start --model-store /models/serve --models fastrcnn=fastrcnn.mar

docker-stop-torch-serve:
	docker stop $$(docker ps -q --filter ancestor=pytorch/torchserve:${TORCHSERVE_VERSION}-gpu)

# Test the torch-serve inference server
docker-test-torch-inference:
	curl http://127.0.0.1:8080/predictions/fastrcnn -T tests/test_data/test.jpg
