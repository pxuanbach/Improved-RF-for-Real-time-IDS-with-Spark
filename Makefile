infras:
	python scripts/update_hosts.py
	docker compose -f ./spark-cluster/docker-compose.yml up --scale spark-worker=3 -d
	bash scripts/run_check.sh

infras-b:
	docker compose -f ./spark-cluster/docker-compose.yml up --scale spark-worker=3 -d --build

clean-infras:
	docker compose -f ./spark-cluster/docker-compose.yml down -v

stats:
	docker stats --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}\t{{.NetIO}}\t{{.BlockIO}}"

install:
	pip install -r requirements.txt
	python scripts/setup_jars.py
