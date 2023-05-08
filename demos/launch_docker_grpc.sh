docker run -it --gpus all -v /home/scott/dev/nos:/nos -v /home/scott/.nos_docker:/app/.nos -p 50051:50051 --rm \
	autonomi-ai/nos:v0.0.0-gpu \
	nos-grpc-server