# bash -c "cd /opt/examples/notebook && jupyter notebook --ip=0.0.0.0 --allow-root"
docker-run-trt-notebook:
	docker run -it --gpus all \
		-p 8888:8888 \
		-v ${PWD}/examples/:/opt/examples \
		torch_tensorrt bash
		