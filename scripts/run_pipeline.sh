#!/bin/bash
# 每日自动 pipeline: sync → convert → train → deploy
# 由 cron 在 14:40 触发，需在 14:50 trade 前完成（通常 ~2 分钟）

set -e

STRATEGY_DIR="$HOME/nas_quant_strategy"
LOG_DIR="$STRATEGY_DIR/logs"
LOG_FILE="$LOG_DIR/pipeline_$(date +%Y%m%d).log"
mkdir -p "$LOG_DIR"

cd "$STRATEGY_DIR"
source .venv/bin/activate

{
echo "$(date '+%Y-%m-%d %H:%M:%S') [pipeline] 开始"

# 1. 从 NAS 同步策略数据
echo "$(date '+%Y-%m-%d %H:%M:%S') [sync] 同步 NAS 数据..."
if [ -f scripts/sync_from_nas.sh ]; then
    bash scripts/sync_from_nas.sh
else
    echo "  跳过同步（脚本不存在）"
fi

# 2. 过滤 + 转换 Qlib 格式
echo "$(date '+%Y-%m-%d %H:%M:%S') [convert] 数据转换..."
python main.py convert

# 3. 训练模型（生成新的 pred_a.pkl）
echo "$(date '+%Y-%m-%d %H:%M:%S') [train] 训练模型..."
python main.py train

# 4. 部署到 trader
echo "$(date '+%Y-%m-%d %H:%M:%S') [deploy] 部署 pred_a.pkl..."
python main.py deploy

echo "$(date '+%Y-%m-%d %H:%M:%S') [pipeline] 完成"
} >> "$LOG_FILE" 2>&1

# 清理 30 天前的日志
find "$LOG_DIR" -name "pipeline_*.log" -mtime +30 -delete
