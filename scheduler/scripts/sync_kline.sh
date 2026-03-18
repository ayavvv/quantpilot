#!/bin/bash
# 从 NAS 同步 Qlib bin 数据到本地 Docker volume
set -e
echo "[$(date '+%Y-%m-%d %H:%M:%S')] sync_qlib: start"

mkdir -p /data/qlib_data

ssh -i /ssh/nas_key -o StrictHostKeyChecking=no theo@192.168.100.131 \
  "cd /volume1/docker/quantpilot/qlib_data && tar cf - calendars instruments features" \
  | tar xf - -C /data/qlib_data/

STOCKS=$(ls /data/qlib_data/features/ | wc -l)
LAST_DATE=$(tail -1 /data/qlib_data/calendars/day.txt 2>/dev/null || echo "unknown")
echo "[$(date '+%Y-%m-%d %H:%M:%S')] sync_qlib: done (stocks=$STOCKS, last_date=$LAST_DATE)"
