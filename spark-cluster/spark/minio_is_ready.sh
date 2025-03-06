#!/bin/bash

MINIO_HEALTH_URL="http://minio:9000/minio/health/live"

echo "Checking MinIO from $(hostname)..."
wget --server-response -q -O /dev/null "$MINIO_HEALTH_URL" 2>&1 | grep "HTTP/" | awk '{print $2}'
echo "Check completed on $(hostname)"
