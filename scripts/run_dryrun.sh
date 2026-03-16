#!/bin/bash
# 每日模拟交易全流程: 同步数据 → 转换 → 执行模拟交易
set -e

cd ~/nas_quant_strategy
source .venv/bin/activate

LOG_DIR="$HOME/nas_quant_strategy/dryrun/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/dryrun_$(date +%Y%m%d).log"

{
  echo "=== Dry Run Pipeline: $(date) ==="

  # 1. 从 NAS 同步最新数据
  echo "[1/3] 同步 NAS 数据..."
  bash scripts/sync_from_nas.sh

  # 2. 转换为 Qlib 格式
  echo "[2/3] 转换 Qlib 格式..."
  python main.py convert

  # 3. 执行模拟交易
  echo "[3/3] 执行模拟交易..."
  python main.py dryrun

  echo "=== Pipeline 完成: $(date) ==="
} 2>&1 | tee "$LOG_FILE"
