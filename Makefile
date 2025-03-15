WORKER_COUNT := 3

infras:
	../Python311/python.exe scripts/update_hosts.py
	docker compose -f ./spark-cluster/docker-compose.yml up --scale spark-worker=$(WORKER_COUNT) -d
	bash scripts/run_check.sh $(WORKER_COUNT)

infras-b:
	docker compose -f ./spark-cluster/docker-compose.yml up --scale spark-worker=3 -d --build

clean-infras:
	docker compose -f ./spark-cluster/docker-compose.yml down -v

stats:
	docker stats --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}\t{{.NetIO}}\t{{.BlockIO}}"

install:
	../Python311/python.exe -m pip install -r requirements.txt
	../Python311/python.exe scripts/setup_jars.py
