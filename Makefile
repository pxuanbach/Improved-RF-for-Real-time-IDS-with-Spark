infras:
	python scripts/update_hosts.py
	docker compose -f ./spark-cluster/docker-compose.yml up --scale spark-worker=3 -d

infras-b:
	docker compose -f ./spark-cluster/docker-compose.yml up --scale spark-worker=3 -d --build

clean-infras:
	docker compose -f ./spark-cluster/docker-compose.yml down -v

install:
	pip install -r requirements.txt
	python scripts/setup_jars.py
