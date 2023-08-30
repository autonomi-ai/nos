SHELL := /bin/bash

docker-compose-upd-discord-bot: 
	pushd ../../ && make docker-build-cpu docker-build-gpu && popd;
	docker compose -f docker-compose.yml up --build