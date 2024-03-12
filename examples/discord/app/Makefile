SHELL := /bin/bash

docker-compose-upd-discord-bot:
	sudo mkdir -p ~/.nosd/volumes
	sudo chown -R $(USER):$(USER) ~/.nosd/volumes
	pushd ../../ && make docker-build-cpu docker-build-gpu && popd;
	docker compose -f docker-compose.yml up --build
