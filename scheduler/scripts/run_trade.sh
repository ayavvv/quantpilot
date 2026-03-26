#!/bin/bash
# 触发 trader 容器执行模拟盘交易
set -e
echo "[$(date '+%Y-%m-%d %H:%M:%S')] run_trade: start"

DOW=$(date +%u)
if [ "$DOW" -gt 5 ]; then
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] run_trade: skip (weekend)"
  exit 0
fi

docker compose -f /compose/docker-compose.mac.yml --project-directory /compose run --rm trader

echo "[$(date '+%Y-%m-%d %H:%M:%S')] run_trade: done"
