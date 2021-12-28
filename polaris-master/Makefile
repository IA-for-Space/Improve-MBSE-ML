SHELL := /bin/bash
docker-compose-start:
	./run_docker_compose.sh

docker-compose-stop:
	docker-compose stop

docker-compose-clean:
	rm -rf docker/influxdb/data/*
	rm -rf docker/grafana/data/*

influxdb-shell:
	docker exec -it influxdb influx

setup:
	( \
		python3 -m venv .venv ; \
		source .venv/bin/activate ; \
		python3 setup.py install ; \
	)
