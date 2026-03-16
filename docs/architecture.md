# QuantPilot Architecture

> Last updated: 2026-03-15

## System Overview

A-share quantitative trading system with flexible deployment:
- **Single-machine**: All services run on one host via Docker
- **Distributed**: NAS for data collection, compute node for everything else

```
┌─────────────────────────────────────────────────────────────────┐
│                      Compute Node                               │
│                                                                 │
│   Futu OpenD ────┐                                              │
│   (:11111)       │                                              │
│                  ▼                                              │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐  │
│   │ Inference │  │ Trader   │  │ Reporter │  │ Weekly Trainer│  │
│   │ (Docker)  │  │ (Docker) │  │ (Docker) │  │ (Docker)     │  │
│   │ 17:00     │  │ 14:50    │  │ 17:30    │  │ Sat 10:00    │  │
│   └─────┬────┘  └────┬─────┘  └────┬─────┘  └──────┬───────┘  │
│         │            │             │                │           │
│         ▼            ▼             ▼                ▼           │
│   $DATA_DIR/                                                    │
│   ├── kline/K_DAY/     (parquet, per-stock)                    │
│   ├── signals/         (csv + pkl, daily predictions)          │
│   ├── models/          (LightGBM .pkl files)                   │
│   └── reports/         (HTML daily reports)                    │
│                                                                 │
│   Observer (Streamlit :8501)                                   │
└────────────────────────┬────────────────────────────────────────┘
                         │ tar + SSH (optional, for distributed)
┌────────────────────────┴────────────────────────────────────────┐
│                    NAS / Data Source (optional)                  │
│                                                                 │
│   ┌──────────────┐                                              │
│   │ Collector    │  baostock daily K-line (5000+ A-shares)     │
│   │ (Docker)     │  16:30 Mon-Fri                              │
│   └──────┬───────┘                                              │
│          ▼                                                      │
│   /data/kline/K_DAY/  (parquet, 15+ years history)             │
└─────────────────────────────────────────────────────────────────┘
```

## Daily Pipeline

```
Time      Event                    Description
────────────────────────────────────────────────────────
14:50     Trader                   Read signals → Futu API trade (simulation)
15:00     Market close             -
16:30     Collector                baostock: collect all A-shares daily K-line
17:00     run_daily.sh             ① Sync kline (if distributed)
                                   ② Docker inference (Qlib + LightGBM)
                                   ③ Docker reporter (daily email)
```

**Weekly:**
- Saturday 10:00: `run_weekly_train.sh` — retrain LightGBM + backtest + deploy

## Inference Pipeline

```
Step 1: Data Conversion (parquet → qlib bin)           ~100s
  - Read parquet files (5000+ stocks)
  - Convert to .bin format (calendar + instruments + features)

Step 2: Model Inference                                 ~140s
  - Load lightgbm_sh_latest.pkl
  - Alpha158Fund handler (~170 features, incl. PE + turnover)
  - 300-day lookback window (memory-optimized)
  - Predict scores for all SH stocks

Step 3: Signal Output
  - signal_YYYYMMDD.csv + pred_sh_daily_YYYYMMDD.pkl
  - Update latest symlinks
```

## Key Technical Parameters

| Parameter | Value |
|-----------|-------|
| Model | LightGBM (MSE, max_depth=6, num_leaves=64, 2000 rounds) |
| Features | Alpha158Fund ~170 (Alpha158 + PE/turnover derivatives) |
| Inference lookback | 300 trading days |
| Training data | 2015-01 ~ latest |
| Universe | A-shares (SH ~2300 stocks) |
| Signal format | `pd.Series`, `MultiIndex(datetime, instrument)`, float64 |
| Data source | baostock (socket, forward-adjusted) |
| Trading API | Futu OpenD (simulation mode by default) |
| Trading params | Top-5, hold bonus 0.05, stop-loss -8% |

## Docker Services

| Service | Platform | Profile | Description |
|---------|----------|---------|-------------|
| collector | linux/arm64 | collector | baostock data collection |
| inference | linux/amd64* | inference | Qlib + LightGBM daily prediction |
| trader | linux/arm64 | trader | Futu API auto trading |
| trainer | linux/amd64* | trainer | Weekly model retraining |
| reporter | linux/arm64 | reporter | Daily HTML email report |
| observer | linux/arm64 | observer | Streamlit monitoring dashboard |

*pyqlib requires amd64 (Rosetta emulation on Apple Silicon)

## Deployment Options

### Single-machine
```bash
docker compose --profile all up -d
```

### Distributed (NAS + Compute)
Configure `.env`:
```
NAS_HOST=192.168.x.x
NAS_USER=your_user
SSH_KEY=~/.ssh/id_ed25519
```

NAS: `docker compose --profile collector up -d`
Compute: cron runs `scripts/run_daily.sh` and `scripts/run_weekly_train.sh`
