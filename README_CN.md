# QuantPilot

A 股量化交易系统 —— 自动化数据采集、模型训练、每日推理、自动交易。

[English](README.md) | 中文

## 系统架构

```
┌─────────────────────────────────────────────────────┐
│                    QuantPilot                        │
│                                                     │
│   Collector ──→ Qlib Data ──→ Trainer ──→ Models    │
│   (baostock)    (bin 直写)    (LightGBM)            │
│       │                                    │        │
│       └─── 直写 qlib bin ──┐               ▼        │
│                            ↓  Inference ──→ 信号文件│
│                        qlib_data  (每日推理) (csv)  │
│                                          │          │
│                         Trader ◄─────────┘          │
│                         (富途 API)                   │
│                                                     │
│   Observer ──→ 仪表板    Reporter ──→ 邮件报告      │
│   (Streamlit)            (SMTP)                     │
└─────────────────────────────────────────────────────┘
```

## 模块说明

| 模块 | 功能 | 定时 |
|------|------|------|
| **collector** | baostock A 股日 K 线采集 (5000+ 股票) | 每日 16:30 |
| **converter** | Qlib 二进制直写 + parquet 迁移工具 | 采集时直写 |
| **strategy** | Qlib + LightGBM + Alpha158Fund 特征 (~170 个) | - |
| **inference** | 每日股票打分预测（宿主机 venv 流水线） | 每日 19:00 |
| **trader** | 富途 OpenD API 自动交易（绑定模拟账户，休市自动预演） | 每日 14:50 |
| **trainer** | 每周模型重训练 + 回测 | 每周六 10:00 |
| **reporter** | HTML 日报邮件 (SMTP) | 每日 19:00 |
| **observer** | Streamlit 系统监控仪表板 | 常驻 |

## 快速开始

### 1. 克隆并配置

```bash
git clone https://github.com/ayavvv/quantpilot.git
cd quantpilot
cp .env.example .env
# 编辑 .env 填入你的配置
```

### 2. 单机部署

```bash
# 启动所有服务
docker compose --profile all up -d

# 或按需启动
docker compose --profile collector up -d      # 数据采集（后台常驻）
docker compose --profile observer up -d       # 监控仪表板 :8501
docker compose run --rm inference             # 单次推理
docker compose run --rm reporter              # 生成日报
```

### 3. 分布式部署（NAS + 计算节点）

**NAS（仅数据采集）：**
```bash
docker compose --profile collector up -d
```

**计算节点（推理、交易、报告）：**
```bash
# 在 .env 中配置 NAS 连接：
# NAS_HOST=192.168.x.x
# NAS_USER=your_user
# SSH_KEY=~/.ssh/id_ed25519

# 交易入口（宿主机 venv；若 OpenD 判断沪深休市会自动切到预演）
./scripts/run_trade.sh

# 每日流水线：等待 NAS + 同步 + 推理 + 日报
./scripts/run_daily.sh

# 或配置 cron：
# 50 14 * * 1-5 /path/to/quantpilot/scripts/run_trade.sh
# 0 19 * * 1-5 /path/to/quantpilot/scripts/run_daily.sh
# 0 10 * * 6   /path/to/quantpilot/scripts/run_weekly_train.sh
```

## 配置说明

所有配置通过环境变量管理，详见 [`.env.example`](.env.example)。

主要参数：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `DATA_DIR` | `~/quantpilot_data` | 数据根目录 |
| `MARKET` | `sh` | 目标市场 (sh/sz/all) |
| `TOP_N` | `5` | 持仓数量 |
| `HOLD_BONUS` | `0.05` | 持仓惯性加分 |
| `STOP_LOSS_PCT` | `-0.08` | 止损阈值 (-8%) |
| `FUTU_SIM_ACC_ID` | `0` | 绑定指定模拟账户；`0` 表示取第一个模拟账户 |
| `DRY_RUN` | `true` | 配置模板默认空跑；生产 `.env` 可改成 `false` |
| `ALLOW_OFF_HOURS_TRADING` | `false` | 是否允许在沪深休市时继续提交订单 |
| `CRON_TIME` | `16:30` | 采集定时 |
| `QLIB_DATA_DIR` | `/qlib_data` | Qlib 二进制数据目录（collector 直写） |

`run_trade.sh` 会先保留外部传入的环境变量，再加载 `.env` 默认值，所以 `DRY_RUN=true ./scripts/run_trade.sh` 不会再被 `.env` 里的配置覆盖。

## 数据流

```
baostock API（免费，socket 协议）
  ↓ 直写 Qlib 格式（无 parquet 中间层）
Qlib 二进制格式（.bin, calendar, instruments, ~30MB）
  ↓ 同步到计算节点（tar+SSH, <5 秒）
Alpha158Fund 特征（~170 个：Alpha158 + PE + 换手率衍生）
  ↓
LightGBM 预测分数
  ↓
信号文件（CSV + pkl MultiIndex Series）
  ↓
富途 OpenD API → 模拟/实盘交易
```

### 直写 Qlib 格式

Collector 采集数据后直接写入 Qlib bin 格式，无需 parquet 中间步骤：

- **传统流程**: baostock → parquet → converter(~100s) → qlib bin
- **当前流程**: baostock → qlib bin（直写，~1-2 秒）

配置 `QLIB_DATA_DIR` 环境变量后自动启用。存量 parquet 数据可通过迁移脚本一次性转换：

```bash
python scripts/migrate_parquet_to_qlib.py --kline-dir /data/kline/K_DAY --qlib-dir /qlib_data
```

## 模型参数

| 参数 | 值 |
|------|-----|
| 算法 | LightGBM (MSE, max_depth=6, num_leaves=64, 2000 轮) |
| 特征 | Alpha158Fund ~170 个 (Alpha158 + PE/换手率衍生) |
| 推理回看 | 300 交易日 (内存优化，避免 OOM) |
| 训练数据 | 2015-01 ~ 最新 |
| 股票池 | 沪市 A 股 (~2300 只) |
| 信号格式 | `pd.Series`, `MultiIndex(datetime, instrument)`, float64 |
| 数据源 | baostock (socket, 前复权) |
| 交易接口 | 富途 OpenD API (默认模拟盘) |
| 交易参数 | Top-5, 持仓加分 0.05, 止损 -8% |

## 交易规则

### 数据规则
- 训练数据使用前复权价格，与实盘一致
- 标签: `Ref($close, -2) / Ref($close, -1) - 1`（次日收益率）
- 训练/验证/测试严格按时间顺序划分

### 回测规则
- 信号日 t 收盘产生 → t+1 收盘买入 → t+2 收盘卖出
- 交易成本: 印花税 (卖 0.05%) + 佣金 (0.025% 双边) + 滑点 (0.1%)
- 双重涨停过滤: 信号日或买入日涨停则跳过
- 等权分配 + 持仓惯性 (减少不必要换手)

### 实盘规则
- 默认模拟盘（代码安全锁）
- 绑定指定模拟账户 (`FUTU_SIM_ACC_ID`)
- 先卖后买，且卖前逐只复核当前真实持仓
- 止损 -8%，立即执行
- 买入价 +1% 滑点确保成交
- 非交易时段或节假日若 OpenD 返回沪深休市，自动切换为预演；仅 `ALLOW_OFF_HOURS_TRADING=true` 时允许盘后测试下单

## 项目结构

```
quantpilot/
├── collector/          # 数据采集 (baostock, futu, yfinance)
├── converter/          # Qlib bin 直写 + 迁移工具
│   ├── incremental.py  # QlibDirectWriter（collector 直接调用）
│   └── loader.py       # 全量转换（迁移/备用）
├── strategy/           # 模型定义 (engine, handler, config)
├── inference/          # 每日推理预测
├── trader/             # 自动交易执行
├── trainer/            # 每周重训练 + 回测
│   └── backtest/       # 回测引擎 + 报告
├── reporter/           # 邮件日报
├── observer/           # Streamlit 监控仪表板
├── scripts/            # Shell 编排脚本
├── docs/               # 文档
├── docker-compose.yml  # Docker 多服务配置
├── .env.example        # 配置模板
└── README.md
```

## 环境要求

- Docker & Docker Compose
- Python 3.10+（pyqlib 兼容性）
- 富途 OpenD（交易用；交易器会读取沪深市场状态决定是否自动预演）
- Mac mini 生产路径为宿主机 `crontab` + `.venv`；Docker trader 仅保留作手动运行
- Apple Silicon 注意: inference/trainer 容器使用 `platform: linux/amd64`（Rosetta 模拟），因为 pyqlib 没有 arm64 wheels

## 许可证

MIT
