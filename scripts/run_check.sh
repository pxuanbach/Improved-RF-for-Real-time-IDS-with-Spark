#!/bin/bash

echo "Waiting for all containers to be ready..."

sleep 5

while ! docker-compose -f ./spark-cluster/docker-compose.yml ps | grep "spark-master" | grep "Up" > /dev/null; do
  echo "Waiting for spark-master..."
  sleep 2
done

while ! curl -s http://localhost:9000 > /dev/null; do
  echo "Waiting for MinIO..."
  sleep 2
done

for i in {1..3}; do
  while ! docker-compose -f ./spark-cluster/docker-compose.yml ps | grep "spark-worker-$i" | grep "Up" > /dev/null; do
    echo "Waiting for spark-worker-$i..."
    sleep 2
  done
done

echo "All containers are ready!"

docker-compose -f ./spark-cluster/docker-compose.yml exec spark-worker /opt/bitnami/spark/minio_is_ready.sh
