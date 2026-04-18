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
| **collector** | NAS 上的 A 股 + 港股日度主采集 | 工作日 18:00 |
| **converter** | Qlib 二进制直写 + parquet 迁移工具 | 采集时直写 |
| **strategy** | Qlib + LightGBM + Alpha158Fund 特征 (~170 个) | - |
| **inference** | 每日股票打分预测（宿主机 venv 流水线） | 每日 19:00 |
| **trader** | 富途 OpenD API 自动交易（绑定模拟账户，休市自动预演） | 每日 14:50 |
| **trainer** | 每周模型重训练 + 回测 | 每周六 10:00 |
| **reporter** | 宿主机原生日报邮件 (SMTP) | 每日 19:00 |
| **observer** | Streamlit 系统监控仪表板 | 常驻 |

## 快速开始

### 1. 克隆并配置

```bash
git clone https://github.com/ayavvv/quantpilot.git
cd quantpilot
cp .env.example .env
# 编辑 .env 填入你的配置
```

### 2. NAS 服务

```bash
docker compose --profile collector up -d      # 数据采集（后台常驻）
docker compose --profile observer up -d       # 监控仪表板 :8501
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

# A 股交易入口（宿主机 venv；若 OpenD 判断沪深休市会自动切到预演）
./scripts/run_trade.sh

# A 股每日流水线：等待 NAS + 同步 + 推理 + 日报
./scripts/run_daily.sh

# 美股 deep-analysis 流水线（与 A 股主链隔离）
./scripts/run_us_daily.sh

# 美股模拟执行（消费 us_trade_plan_latest.json）
DRY_RUN=true ./scripts/run_us_trade.sh

# 或配置 cron：
# 50 14 * * 1-5 /path/to/quantpilot/scripts/run_trade.sh
# 0 19 * * 1-5 /path/to/quantpilot/scripts/run_daily.sh
# 0 10 * * 6   /path/to/quantpilot/scripts/run_weekly_train.sh
```

### 4. 美股 deep-analysis 流水线

美股链路刻意与现有 A 股生产链路隔离：
- 不读取也不覆盖 `pred_sh_latest.pkl`
- 不改动 `scripts/run_trade.sh`
- 输出写到 `signals/us/` 下的独立文件

流程：

```text
S&P 500 成分股（或 US_TARGET_CODES 覆盖）
  -> 价格 / 流动性过滤
  -> 对候选股逐个跑 deep-analysis
  -> 产出 us_trade_plan_latest.json
  -> trader.trade_us_daily 执行
```

常用联调命令：

```bash
# 先用少量 ticker 验证整条链路
US_TARGET_CODES=US.AAPL,US.MSFT US_ANALYSIS_TOP_K=2 ./scripts/run_us_daily.sh

# 生产默认：每天 Top 20 分析、并发 10、单票 1 小时、最终 Top 5 模拟持仓
./scripts/run_us_daily.sh

# 如果当前机器不是通过 futu-opend:11111 暴露 OpenD，显式覆盖地址和 RSA key
FUTU_HOST=192.168.100.248 FUTU_PORT=11111 FUTU_RSA_KEY=/path/to/futu_rsa_1024.pem US_TARGET_CODES=US.AAPL,US.MSFT US_ANALYSIS_TOP_K=2 ./scripts/run_us_daily.sh

# 只做执行预演
DRY_RUN=true ./scripts/run_us_trade.sh
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
├── docker-compose.yml  # NAS Docker 服务（collector / observer / 手动任务）
├── docker-compose.mac.yml  # Mac mini 手动 Docker 任务（调试用）
├── .env.example        # 配置模板
└── README.md
```

## 环境要求

- Docker & Docker Compose
- Python 3.10+（pyqlib 兼容性）
- 富途 OpenD（交易用；交易器会读取沪深市场状态决定是否自动预演）
- Mac mini 生产路径为宿主机 `crontab` + `.venv`；`docker-compose.mac.yml` 仅保留作手动隔离调试
- Apple Silicon 注意: inference/trainer 容器使用 `platform: linux/amd64`（Rosetta 模拟），因为 pyqlib 没有 arm64 wheels

## 许可证

MIT
