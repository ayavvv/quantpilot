# QuantPilot

English | [中文](README_CN.md)

A-share quantitative trading system with automated data collection, model training, daily inference, and trade execution.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    QuantPilot                        │
│                                                     │
│   Collector ──→ Qlib Data ──→ Trainer ──→ Models    │
│   (baostock)    (bin direct)  (LightGBM)            │
│                                           │         │
│                              Inference ──→ Signals  │
│                              (daily)      (csv/pkl) │
│                                             │       │
│                            Trader ◄─────────┘       │
│                            (Futu API)               │
│                                                     │
│   Observer ──→ Dashboard    Reporter ──→ Email      │
│   (Streamlit)               (SMTP)                  │
└─────────────────────────────────────────────────────┘
```

## Modules

| Module | Description | Schedule |
|--------|-------------|----------|
| **collector** | A-share daily K-line data collection via baostock | Daily 16:30 |
| **converter** | Qlib binary direct writer + parquet migration | Part of collector |
| **strategy** | Qlib-based LightGBM model with Alpha158Fund features (~170) | - |
| **inference** | Daily stock score prediction | Daily 17:00 |
| **trader** | Auto trading via Futu OpenD API (simulation mode) | Daily 14:50 |
| **trainer** | Weekly model retraining + backtest | Saturday 10:00 |
| **reporter** | HTML daily report via SMTP email | Daily 17:30 |
| **observer** | Streamlit monitoring dashboard | Always-on |

## Quick Start

### 1. Clone and configure

```bash
git clone https://github.com/your-username/quantpilot.git
cd quantpilot
cp .env.example .env
# Edit .env with your settings
```

### 2. Single-machine deployment

```bash
# Start all services
docker compose --profile all up -d

# Or start specific services
docker compose --profile collector up -d      # Data collection daemon
docker compose --profile observer up -d       # Monitoring at :8501
docker compose run --rm inference             # One-shot inference
docker compose run --rm reporter              # Generate report
```

### 3. Distributed deployment (NAS + compute node)

**NAS (data collection only):**
```bash
docker compose --profile collector up -d
```

**Compute node (inference, trading, reporting):**
```bash
# Configure NAS connection in .env:
# NAS_HOST=192.168.x.x
# NAS_USER=your_user
# SSH_KEY=~/.ssh/id_ed25519

# Daily pipeline (sync + inference + report)
./scripts/run_daily.sh

# Or via cron:
# 0 17 * * 1-5 /path/to/quantpilot/scripts/run_daily.sh
# 0 10 * * 6   /path/to/quantpilot/scripts/run_weekly_train.sh
```

## Configuration

All configuration is via environment variables. See [`.env.example`](.env.example) for the full list.

Key settings:

| Variable | Default | Description |
|----------|---------|-------------|
| `DATA_DIR` | `~/quantpilot_data` | Root data directory |
| `MARKET` | `sh` | Target market (sh/sz/all) |
| `TOP_N` | `5` | Number of positions |
| `HOLD_BONUS` | `0.05` | Hold inertia score bonus |
| `STOP_LOSS_PCT` | `-0.08` | Stop-loss threshold (-8%) |
| `DRY_RUN` | `true` | Dry-run mode (no real trades) |
| `CRON_TIME` | `16:30` | Collector schedule time |

## Data Flow

```
baostock API (free, socket-based)
  ↓ direct write (no parquet intermediate)
Qlib binary format (.bin, calendar, instruments, ~30MB)
  ↓ sync to compute node (tar+SSH, <5s)
Alpha158Fund handler (~170 features: Alpha158 + PE + turnover)
  ↓
LightGBM prediction scores
  ↓
Signal files (CSV + pkl with MultiIndex Series)
  ↓
Futu OpenD API → simulated/real trades
```

## Model

- **Algorithm**: LightGBM (MSE loss, max_depth=6, num_leaves=64, 2000 rounds)
- **Features**: Alpha158Fund (~170 features including PE and turnover derivatives)
- **Training data**: 2015-01 to latest
- **Inference lookback**: 300 trading days (memory-optimized)
- **Retraining**: Weekly (Saturday)

## Trading Rules

- **Universe**: Shanghai A-shares (~2300 stocks)
- **Selection**: Top-N by model score with hold inertia bonus
- **Filters**: Dual limit-up filter (signal day + buy day)
- **Risk**: -8% stop-loss per position
- **Execution**: Equal-weight, sell-first-then-buy via Futu API
- **Mode**: Simulation by default (safety lock in code)

## Project Structure

```
quantpilot/
├── collector/          # Data collection (baostock, futu, yfinance)
├── converter/          # Parquet → Qlib bin format
├── strategy/           # Model definition (engine, handler, config)
├── inference/          # Daily prediction pipeline
├── trader/             # Auto trading execution
├── trainer/            # Weekly retraining + backtest
│   └── backtest/       # Backtest engine + reporting
├── reporter/           # Email report generation
├── observer/           # Streamlit monitoring dashboard
├── scripts/            # Shell orchestration scripts
├── docs/               # Documentation
├── docker-compose.yml  # Multi-service Docker config
├── .env.example        # Configuration template
└── README.md
```

## Requirements

- Docker & Docker Compose
- Python 3.10+ (for pyqlib compatibility)
- Futu OpenD (for trading, optional for inference-only)
- Apple Silicon note: inference/trainer containers use `platform: linux/amd64` (Rosetta) because pyqlib lacks arm64 wheels

## License

MIT
