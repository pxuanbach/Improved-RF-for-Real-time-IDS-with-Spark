WORKER_COUNT := 4

infras:
	python scripts/update_hosts.py
	docker compose -f ./spark-cluster/docker-compose.yml up --scale spark-worker=$(WORKER_COUNT) -d
	bash scripts/run_check.sh $(WORKER_COUNT)

infras-b:
	docker compose -f ./spark-cluster/docker-compose.yml up --scale spark-worker=3 -d --build

clean-infras:
	docker compose -f ./spark-cluster/docker-compose.yml down -v

stats:
	docker stats --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}\t{{.NetIO}}\t{{.BlockIO}}"

install:
	python -m pip install -r requirements.txt
	python scripts/setup_jars.py

impro-rf-pipeline:
	cd improved-ids && \
	python ./run.py

monitor-app:
	cd improved-ids && \
	uvicorn api.main:app --reload
