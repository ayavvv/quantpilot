# QuantPilot 架构文档

> 更新时间: 2026-03-17

## 系统概览

全 A 股量化选股 + 模拟盘自动交易系统。单一 Git 仓库，全 Docker 化部署，双机分布式架构。

```
┌─ Mac mini (计算节点) ─────────────────────────────────────────────┐
│                                                                    │
│   Futu OpenD (原生 GUI)                                            │
│   127.0.0.1:11111                                                  │
│       ▲                                                            │
│       │ host.docker.internal                                       │
│       │                                                            │
│   ┌───┴──────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │
│   │ Trader   │  │Inference │  │ Reporter │  │   Scheduler      │  │
│   │ 14:50    │  │ 17:00    │  │ 17:00    │  │ (crond, 常驻)    │  │
│   └──────────┘  └──────────┘  └──────────┘  └──────────────────┘  │
│       │              │             │                               │
│       ▼              ▼             ▼                               │
│   ~/quantpilot_data/                                               │
│   ├── qlib_data/     (Qlib bin, 从 NAS 同步)                      │
│   ├── signals/       (pred_a.pkl, signal_*.csv)                    │
│   ├── models/        (lightgbm_sh_latest.pkl)                      │
│   └── reports/       (HTML 日报)                                   │
│                                                                    │
└────────────────────────┬───────────────────────────────────────────┘
                         │ SSH + tar (14:40 自动同步)
┌─ Synology NAS (数据节点) ─────────────────────────────────────────┐
│                                                                    │
│   ┌──────────────────┐    ┌──────────────┐                         │
│   │ Collector        │    │ Observer     │                         │
│   │ (APScheduler)    │    │ (Streamlit)  │                         │
│   │ 16:30 / 07:00    │    │ :8501        │                         │
│   └────────┬─────────┘    └──────────────┘                         │
│            ▼                                                       │
│   /volume1/docker/quantpilot/                                      │
│   ├── qlib_data/     (Qlib bin, 源数据)                            │
│   │   ├── calendars/day.txt                                        │
│   │   ├── instruments/{all,sh,sz,hk}.txt                           │
│   │   └── features/{code}/{field}.day.bin                          │
│   └── data/          (辅助: 基本面, 做空, 行业映射)                │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

## 数据格式

**核心数据全部使用 Qlib bin 格式**，无 parquet 中间层。

| 数据 | 格式 | 写入方 | 读取方 |
|------|------|--------|--------|
| A 股日 K 线 | Qlib bin | Collector (QlibDirectWriter) | Inference, Trader |
| HK 股日 K 线 | Qlib bin | Collector (QlibDirectWriter) | Inference |
| US ETF / 宏观 | Qlib bin | Collector (QlibDirectWriter) | Inference |
| 模型信号 | pkl + csv | Inference | Trader |
| 基本面快照 | parquet (辅助) | Collector | — |
| 分钟 K 线 | parquet (辅助) | Collector | — |

Qlib bin 文件结构:
```
qlib_data/
├── calendars/day.txt              # 交易日历 (每行一个日期)
├── instruments/all.txt            # 股票列表 (code\tstart\tend)
└── features/{code}/               # 每只股票一个目录
    ├── open.day.bin               # [start_index(f32), val1, val2, ...]
    ├── close.day.bin
    ├── high.day.bin
    ├── low.day.bin
    ├── volume.day.bin
    ├── amount.day.bin
    ├── change_rate.day.bin
    └── ...
```

## 每日流程

```
时间       事件              描述
──────────────────────────────────────────────────────────────────
07:00      Collector         US 股 / ETF K 线采集 (Futu + YFinance)
07:30      Collector         宏观指标采集 (VIX, DXY, TNX, HSI)
09:30      A 股开盘          —
14:40      Scheduler         从 NAS 同步 Qlib bin 数据到本地
14:50      Trader            读信号 → 涨停过滤 → 先卖后买 → Futu 模拟盘下单
15:00      A 股收盘          —
16:30      Collector         A 股日 K 线采集 (baostock, 5000+ 只)
                             HK 股采集 (Futu, 含基本面/做空)
17:00      Scheduler         触发 Inference + Reporter
           Inference         验证 Qlib 数据 → LightGBM 预测 → 输出 pred_a.pkl
           Reporter          生成 HTML 日报 → 邮件/本地保存
```

**周任务:**
- 周一 08:00: Collector 刷新行业板块映射

## Docker 服务

### NAS 端 (`docker-compose.yml`)

| 服务 | Profile | 功能 | 调度 |
|------|---------|------|------|
| collector | collector | baostock + Futu + YFinance 数据采集, 直写 Qlib bin | APScheduler (容器内) |
| observer | observer | Streamlit 监控看板 (:8501) | 常驻 |

```bash
# NAS 启动
docker compose --profile collector --profile observer up -d
```

### Mac mini 端 (`docker-compose.mac.yml`)

| 服务 | 功能 | 调度 |
|------|------|------|
| scheduler | crond 调度器, 触发其他服务 | 常驻 |
| trader | 模拟盘自动交易 (Futu OpenD) | 14:50 工作日 |
| inference | Qlib + LightGBM 模型推理 (x86 Rosetta) | 17:00 工作日 |
| reporter | HTML 日报 + 邮件 | 17:00 工作日 |

```bash
# Mac mini 启动
cd ~/quantpilot
docker compose -f docker-compose.mac.yml up -d scheduler
# 手动触发交易
docker compose -f docker-compose.mac.yml run --rm trader
# 手动触发推理
docker compose -f docker-compose.mac.yml run --rm inference
```

## 模型参数

| 参数 | 值 |
|------|-----|
| 模型 | LightGBM (MSE) |
| 因子 | Alpha158A (173 个): Alpha158 + 基本面(PE/PB/换手率) + 宏观(VIX/DXY/TNX/SPY) |
| LGB 参数 | lr=0.05, depth=7, leaves=127, l1=10, l2=20, rounds=500, early_stop=50 |
| 训练区间 | Train 2018-2024H1, Valid 2024H2-2025H1, Test 2025H2+ |
| 最新指标 | IC=0.035, ICIR=0.389 |
| 回测 (2025H2) | 年化超额 129%, Sharpe 3.53, 最大回撤 -12.7% |

## 交易参数

| 参数 | 值 | 说明 |
|------|-----|------|
| TOP_N | 5 | 持仓数量 |
| HOLD_BONUS | 0.05 | 持仓惯性加分 |
| STOP_LOSS_PCT | -8% | 止损线 |
| POSITION_RATIO | 95% | 仓位比例 |
| BUY_SLIPPAGE | +1% | 买入滑点 (确保成交) |
| SELL_SLIPPAGE | -1% | 卖出滑点 |
| 安全锁 | TrdEnv.SIMULATE | 硬编码 + assert, 禁止实盘 |

### 涨停过滤规则

| 板块 | 涨停阈值 | 最小手数 |
|------|----------|----------|
| 主板 (60xxxx) | ≥ 9.5% | 100 股 |
| 创业板 (300xxx) | ≥ 19.5% | 100 股 |
| 科创板 (688xxx) | ≥ 19.5% | 200 股 |

双重过滤: 信号日涨停 + 买入日涨停, 两者都排除。

## 仓库结构

```
quantpilot/                          # 单一 Git monorepo
├── collector/                       # NAS 数据采集器
│   ├── Dockerfile
│   ├── scheduler.py                 # APScheduler 调度 + 数据采集
│   ├── baostock_client.py           # A 股数据源
│   ├── futu_client.py               # 港股/美股数据源
│   ├── yf_client.py                 # YFinance 备用源
│   ├── db_engine.py                 # 辅助数据存储 (分钟线/元数据)
│   └── config.py                    # Pydantic 配置
│
├── converter/
│   └── incremental.py               # QlibDirectWriter + QlibBinReader
│
├── strategy/                        # 策略引擎 (Qlib)
│   ├── engine.py                    # 训练/推理引擎 (A 股 + 港股)
│   ├── backtest.py                  # TopkDropout 回测
│   ├── stock_filter.py              # ST/次新/低流动性过滤
│   ├── paper_trader.py              # 纸面模拟交易
│   ├── index_predictor.py           # 恒指/科技指数预测
│   ├── config_a.yaml                # A 股模型配置
│   └── config_hk.yaml               # 港股模型配置
│
├── inference/                       # 每日推理服务
│   ├── Dockerfile
│   └── run_daily.py                 # 验证数据 → 模型预测 → 输出信号
│
├── trader/                          # 自动交易服务
│   ├── Dockerfile
│   └── trade_daily.py               # 信号 → 涨停过滤 → Futu 下单
│
├── trainer/                         # 周训练服务
│   ├── Dockerfile
│   ├── weekly_train.py
│   └── backtest/                    # 回测框架
│
├── reporter/                        # 日报服务
│   ├── Dockerfile
│   └── send_report.py               # HTML 报告 + SMTP
│
├── observer/                        # 监控看板
│   ├── Dockerfile
│   └── app.py                       # Streamlit dashboard
│
├── scheduler/                       # Mac mini Docker 调度器
│   ├── Dockerfile
│   ├── crontab                      # 定时任务配置
│   └── scripts/                     # sync / trade / daily 脚本
│
├── dashboard/                       # Streamlit 策略看板
│   └── app.py
│
├── scripts/                         # 运维脚本
│   ├── sync_data.sh                 # NAS → 本地 Qlib 数据同步
│   ├── run_pipeline.sh              # 完整训练流水线
│   └── migrate_parquet_to_qlib.py   # 一次性迁移工具
│
├── docker-compose.yml               # NAS 全服务 (profile-based)
├── docker-compose.mac.yml           # Mac mini 服务
├── strategy_cli.py                  # CLI: train / backtest / dryrun
├── requirements.txt                 # Python 基础依赖
└── requirements-qlib.txt            # Qlib 专用依赖
```

## 数据源

| 数据源 | 协议 | 覆盖范围 | 用途 |
|--------|------|----------|------|
| baostock | Socket | 全 A 股 (~5200 只) | 日 K 线 (主力数据源) |
| Futu OpenD | TCP | 港股 + 美股 | K 线 + 基本面 + 做空 + 行业 |
| YFinance | HTTP | 美股 ETF + 宏观指数 | YINN/CQQQ/KWEB/FXI + VIX/DXY/TNX |

## A 股过滤规则

模型训练和交易前, 对股票池进行过滤:

| 规则 | 条件 | 效果 |
|------|------|------|
| ST | 名称含 "ST" | 剔除 |
| 次新股 | 交易日 < 252 天 | 剔除 |
| 低流动性 | 近 60 日日均成交额 < 5000 万 | 剔除 |

典型结果: 5200 只 A 股 → ~4700 只通过

## Git 远程

| Remote | URL | 用途 |
|--------|-----|------|
| origin | `ssh://theo@NAS/quantpilot.git` | NAS Gitea 备份 |
| github | `https://github.com/ayavvv/quantpilot` | GitHub 备份 |

## 环境依赖

| 组件 | 版本 | 说明 |
|------|------|------|
| Python | 3.12 | 策略/推理/交易 |
| Docker | 28.x | OrbStack (Mac) / Synology (NAS) |
| Futu OpenD | 10.0.6018 | Mac mini GUI 版, 127.0.0.1:11111 |
| pyqlib | ≥ 0.9.0 | 仅 amd64 (Mac 上需 Rosetta) |
| LightGBM | ≥ 3.3 | 模型训练/推理 |
