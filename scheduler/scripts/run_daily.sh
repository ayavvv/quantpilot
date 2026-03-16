#!/bin/bash
# 每日推理 + 日报
set -e
echo ""
echo "=========================================="
echo "QuantPilot Daily Pipeline — $(date +%Y-%m-%d)"
echo "=========================================="

DOW=$(date +%u)
if [ "$DOW" -gt 5 ]; then
  echo "skip (weekend)"
  exit 0
fi

# 1. 同步最新数据
echo "[$(date '+%H:%M:%S')] Step 1: sync kline from NAS..."
/scripts/sync_kline.sh

# 2. 推理
echo "[$(date '+%H:%M:%S')] Step 2: inference..."
docker compose -f /compose/docker-compose.yml --project-directory /compose run --rm inference

# 3. 日报
echo "[$(date '+%H:%M:%S')] Step 3: reporter..."
docker compose -f /compose/docker-compose.yml --project-directory /compose run --rm reporter

echo "[$(date '+%H:%M:%S')] Daily pipeline finished"
