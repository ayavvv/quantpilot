"""量化策略 CLI 入口。需 Python 3.12 + pyqlib。"""
from pathlib import Path
import subprocess
import sys

try:
    from dotenv import load_dotenv
    _root = Path(__file__).resolve().parent
    load_dotenv(_root / ".env")
    if not (_root / ".env").exists():
        load_dotenv(_root / ".env.example")
except ImportError:
    _root = Path(__file__).resolve().parent


def _check_qlib() -> None:
    try:
        from strategy.engine import StrategyEngine  # noqa: F401
    except ImportError:
        print("训练/看板依赖 pyqlib，请使用 Python 3.12 并执行：")
        print("  pip install -r requirements.txt -r requirements-qlib.txt")
        sys.exit(1)


# ── 训练 ─────────────────────────────────────────────────────────────────────

def train(experiment_name: str = "a_share_lgbm", save_dir: str | None = None) -> None:
    """训练全 A 股 LightGBM 模型"""
    _check_qlib()
    from strategy.engine import StrategyEngine
    print("=" * 60)
    print("训练全 A 股选股模型 (LightGBM + Alpha158A)")
    print("=" * 60)
    engine = StrategyEngine()
    metrics = engine.train_a_model(
        experiment_name=experiment_name,
        save_dir=Path(save_dir) if save_dir else None,
    )
    print("训练完成:", metrics)


def train_hk(experiment_name: str = "hk_lgbm", save_dir: str | None = None) -> None:
    """训练港股选股 LightGBM 模型"""
    _check_qlib()
    from strategy.engine import StrategyEngine
    print("=" * 50)
    print("策略引擎：训练港股选股模型")
    print("=" * 50)
    engine = StrategyEngine.for_hk()
    metrics = engine.train_hk_model(
        experiment_name=experiment_name,
        save_dir=Path(save_dir) if save_dir else None,
    )
    print("训练完成:", metrics)


# ── 回测 ─────────────────────────────────────────────────────────────────────

def backtest(topk: int = 10, n_drop: int = 2) -> None:
    """对已训练 A 股模型执行 TopkDropout 回测"""
    _check_qlib()
    from strategy.backtest import run_backtest
    print("=" * 60)
    print(f"A 股回测：Top-{topk}，每日换 {n_drop} 只")
    print("=" * 60)
    run_backtest(topk=topk, n_drop=n_drop)


def backtest_hk(topk: int = 10, n_drop: int = 2) -> None:
    """对已训练港股模型执行回测"""
    _check_qlib()
    from strategy.backtest import run_hk_backtest
    print("=" * 50)
    print(f"港股回测：Top-{topk}，每日换 {n_drop} 只")
    print("=" * 50)
    run_hk_backtest(topk=topk, n_drop=n_drop)


# ── 指数预测 ─────────────────────────────────────────────────────────────────

def train_index() -> None:
    """训练恒指/恒生科技指数时序预测模型"""
    from strategy.index_predictor import train_all_index_models
    print("=" * 50)
    print("指数预测模型训练（恒指 + 恒生科技）")
    print("=" * 50)
    results = train_all_index_models()
    for r in results:
        print(f"  {r['code']}: IC={r['test_ic']:.4f}  方向准确率={r['test_dir_acc']:.2%}")


def predict_index() -> None:
    """预测恒指/恒生科技明日走势"""
    from strategy.index_predictor import predict_all_indexes
    print("=" * 50)
    print("指数明日走势预测")
    print("=" * 50)
    predict_all_indexes()


# ── 看板 ─────────────────────────────────────────────────────────────────────

def dashboard() -> None:
    """启动 Streamlit 量化看板"""
    _check_qlib()
    app_path = _root / "dashboard" / "app.py"
    print("启动看板:", app_path, "（Ctrl+C 退出）")
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(app_path), "--server.headless", "true"],
        cwd=str(_root),
    )


def dryrun(topk: int = 10, n_drop: int = 2, initial_cash: float = 1_000_000) -> None:
    """执行一次模拟交易（Dry Run）"""
    _check_qlib()
    from strategy.paper_trader import PaperTrader
    print("=" * 50)
    print("模拟交易 Dry Run")
    print("=" * 50)
    trader = PaperTrader(
        topk=topk, n_drop=n_drop,
        initial_cash=initial_cash,
        hk_mode=False,
    )
    result = trader.execute_daily()
    report = trader.get_daily_report(result)
    print(report)
    report_dir = _root / "dryrun"
    report_path = report_dir / f"report_{result['date']}.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(f"报告已保存: {report_path}")


def dryrun_status() -> None:
    """查看模拟交易组合状态"""
    _check_qlib()
    from strategy.paper_trader import PaperTrader
    trader = PaperTrader(hk_mode=False)
    print(trader.get_portfolio_summary())


def pipeline() -> None:
    """完整流水线：sync -> train -> backtest -> deploy"""
    print("=" * 60)
    print("全 A 股量化 Pipeline")
    print("=" * 60)
    # 1. 同步 Qlib 数据
    print("\n[1/4] 从 NAS 同步 Qlib 数据...")
    sync_script = _root / "scripts" / "sync_data.sh"
    if sync_script.exists():
        subprocess.run(["bash", str(sync_script)], check=True)
    else:
        print("  跳过同步（sync_data.sh 不存在）")
    # 2. 训练
    print("\n[2/4] 训练模型...")
    _check_qlib()
    train()
    # 3. 回测
    print("\n[3/4] 回测验证...")
    backtest()
    # 4. 完成
    print("\n" + "=" * 60)
    print("Pipeline 完成!")
    print("=" * 60)


# ── 入口 ─────────────────────────────────────────────────────────────────────

COMMANDS = {
    "pipeline":      (pipeline,      "完整流水线: sync -> train -> backtest"),
    "train":         (train,         "训练全 A 股 LightGBM 模型"),
    "train-hk":      (train_hk,      "训练港股选股模型"),
    "backtest":      (backtest,      "A股 TopkDropout 回测"),
    "backtest-hk":   (backtest_hk,   "港股 TopkDropout 回测"),
    "train-index":   (train_index,   "训练恒指/科技指数预测模型"),
    "predict-index": (predict_index, "预测恒指/科技指数明日走势"),
    "dryrun":        (dryrun,        "执行模拟交易 (TopkDropout Dry Run)"),
    "dryrun-status": (dryrun_status, "查看模拟交易组合当前状态"),
    "dashboard":     (dashboard,     "启动 Streamlit 看板"),
}


def main() -> None:
    if len(sys.argv) <= 1 or sys.argv[1] in ("-h", "--help"):
        print("用法: python strategy_cli.py <命令>\n")
        for cmd, (_, desc) in COMMANDS.items():
            print(f"  {cmd:<18} {desc}")
        print()
        return

    cmd = sys.argv[1].lower()
    if cmd not in COMMANDS:
        print(f"未知命令: {cmd}")
        print("可用命令:", ", ".join(COMMANDS))
        sys.exit(1)

    COMMANDS[cmd][0]()


if __name__ == "__main__":
    main()
