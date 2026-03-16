#!/bin/bash
# 从 NAS 同步全 A 股 K 线数据 + 基本面到本地
set -e

NAS_USER="theo"
NAS_IP="100.81.131.24"
NAS_DATA="/volume1/docker/nas_quant/data/"
LOCAL_DATA="$HOME/nas_quant_strategy/data/"
SSH_KEY="$HOME/.ssh/nas_key"
RSYNC_SSH="ssh -i $SSH_KEY"

echo "[sync] 开始同步 NAS 数据: $(date)"

# 同步 K 线日线数据（全部：HK + SH + SZ + US + MACRO）
rsync -avz --delete \
  --rsync-path=/usr/bin/rsync \
  -e "$RSYNC_SSH" \
  "${NAS_USER}@${NAS_IP}:${NAS_DATA}kline/K_DAY/" \
  "${LOCAL_DATA}kline/K_DAY/" \
  --exclude="*.tmp" \
  --quiet

# 同步基本面数据
rsync -avz \
  --rsync-path=/usr/bin/rsync \
  -e "$RSYNC_SSH" \
  "${NAS_USER}@${NAS_IP}:${NAS_DATA}fundamentals/" \
  "${LOCAL_DATA}fundamentals/" \
  --exclude="*.tmp" \
  --quiet 2>/dev/null || true

SH_COUNT=$(ls "${LOCAL_DATA}kline/K_DAY/" | grep -c '^SH\.' || true)
SZ_COUNT=$(ls "${LOCAL_DATA}kline/K_DAY/" | grep -c '^SZ\.' || true)
HK_COUNT=$(ls "${LOCAL_DATA}kline/K_DAY/" | grep -c '^HK\.' || true)
echo "[sync] 同步完成: SH=${SH_COUNT} SZ=${SZ_COUNT} HK=${HK_COUNT} $(date)"
