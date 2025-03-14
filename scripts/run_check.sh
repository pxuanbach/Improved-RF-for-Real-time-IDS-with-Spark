#!/bin/bash

WORKER_COUNT=${1:-3}

echo "Waiting for all containers to be ready..."
sleep 5

while ! docker-compose -f ./spark-cluster/docker-compose.yml ps | grep -E "spark-master" | grep -E "Up|running" > /dev/null; do
  echo "Waiting for spark-master..."
  sleep 2
done

while ! curl -s http://localhost:9000 > /dev/null; do
  echo "Waiting for MinIO..."
  sleep 2
done

for i in $(seq 1 $WORKER_COUNT); do
  while ! docker-compose -f ./spark-cluster/docker-compose.yml ps | grep -E "spark-worker-$i" | grep -E "Up|running" > /dev/null; do
    echo "Waiting for spark-worker-$i..."
    sleep 2
  done
done

echo "All containers are ready!"

if [[ "$(uname -s)" =~ "MINGW" ]]; then
  echo "Detected Windows (Git Bash) - Adjusting command format..."
  docker-compose -f ./spark-cluster/docker-compose.yml exec spark-worker bash -c "//opt//bitnami//spark//minio_is_ready.sh"
else
  docker-compose -f ./spark-cluster/docker-compose.yml exec spark-worker /opt/bitnami/spark/minio_is_ready.sh
fi
