infras:
	docker compose -f ./spark-cluster/docker-compose.yml up --scale spark-worker=3 -d
