"""量化策略入口。需 Python 3.12 + pyqlib。"""
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
    pass

from strategy.loader import convert_data


def _check_qlib() -> None:
    try:
        from strategy.engine import StrategyEngine  # noqa: F401
    except ImportError:
        print("训练/看板依赖 pyqlib，请使用 Python 3.12 并执行：")
        print("  pip install -r requirements.txt -r requirements-qlib.txt")
        sys.exit(1)


# ── A股流程 ──────────────────────────────────────────────────────────────────

def convert() -> None:
    """数据转换：Parquet -> Qlib Bin（全 A 股，含过滤）"""
    from strategy.stock_filter import filter_stock_universe
    from converter.loader import _get_data_source

    print("=" * 60)
    print("A 股数据转换（含股票池过滤）")
    print("=" * 60)

    data_source = _get_data_source()
    print(f"数据源: {data_source}")
    print("过滤规则: 剔除 ST、次新股(<1年)、日均成交额<5000万")

    passed, rejected = filter_stock_universe(data_source)
    print(f"通过过滤: {len(passed)} 只")
    print(f"被剔除: {len(rejected)} 只")

    # 打印剔除统计
    st_count = sum(1 for r in rejected.values() if r.startswith("ST"))
    ipo_count = sum(1 for r in rejected.values() if r.startswith("次新"))
    vol_count = sum(1 for r in rejected.values() if r.startswith("低流动"))
    other_count = len(rejected) - st_count - ipo_count - vol_count
    print(f"  ST: {st_count}, 次新股: {ipo_count}, 低流动性: {vol_count}, 其他: {other_count}")

    convert_data(whitelist=set(passed))


def convert_hk() -> None:
    """数据转换：Parquet -> Qlib Bin（港股）"""
    import os
    from converter.loader import convert_data as _convert
    print("=" * 50)
    print("Qlib Parquet -> Bin 数据转换（港股）")
    print("=" * 50)
    data_src = os.environ.get("DATA_SOURCE", str(_root / "data/kline/K_DAY"))
    hk_tmp = _root / "data" / "kline" / "K_DAY_HK_only"
    hk_tmp.mkdir(parents=True, exist_ok=True)
    import shutil
    src = Path(data_src)
    for d in src.iterdir():
        if d.name.startswith("HK.") and d.is_dir():
            dst = hk_tmp / d.name
            if not dst.exists():
                shutil.copytree(d, dst)
    os.environ["DATA_SOURCE"] = str(hk_tmp)
    from pathlib import Path as P
    import importlib, converter.loader as cl
    orig = cl._get_output_dir
    cl._get_output_dir = lambda: P("~/.qlib/qlib_data/hk_quant_data").expanduser().resolve()
    try:
        _convert()
    finally:
        cl._get_output_dir = orig
        os.environ["DATA_SOURCE"] = data_src


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
    report_dir = Path(__file__).resolve().parent / "dryrun"
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


def deploy() -> None:
    """部署：将 pred_a.pkl 复制到 trader 目录"""
    import shutil
    src = _root / "models" / "pred_a.pkl"
    if not src.exists():
        src = _root / "models" / "pred.pkl"
    if not src.exists():
        print("未找到预测文件，请先运行 train")
        return
    dst = Path.home() / "nas_quant_trader" / "models" / "pred_a.pkl"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"已部署: {src} -> {dst}")
    print(f"文件大小: {dst.stat().st_size / 1024:.1f} KB")


def pipeline() -> None:
    """完整流水线：sync -> filter -> convert -> train -> backtest -> deploy"""
    print("=" * 60)
    print("全 A 股量化 Pipeline")
    print("=" * 60)
    # 1. 同步
    print("\n[1/5] 从 NAS 同步数据...")
    sync_script = _root / "scripts" / "sync_from_nas.sh"
    if sync_script.exists():
        subprocess.run(["bash", str(sync_script)], check=True)
    else:
        print("  跳过同步（sync_from_nas.sh 不存在）")
    # 2. 过滤 + 转换
    print("\n[2/5] 过滤 + Qlib 转换...")
    convert()
    # 3. 训练
    print("\n[3/5] 训练模型...")
    _check_qlib()
    train()
    # 4. 回测
    print("\n[4/5] 回测验证...")
    backtest()
    # 5. 部署
    print("\n[5/5] 部署到 trader...")
    deploy()
    print("\n" + "=" * 60)
    print("Pipeline 完成!")
    print("=" * 60)


def run() -> None:
    """完整流程：convert -> train -> dashboard"""
    _check_qlib()
    convert(); print()
    train(); print()
    dashboard()


# ── 入口 ─────────────────────────────────────────────────────────────────────

COMMANDS = {
    "pipeline":      (pipeline,      "完整流水线: sync -> filter -> convert -> train -> backtest -> deploy"),
    "run":           (run,           "完整 A股流程：转换 -> 训练 -> 看板"),
    "convert":       (convert,       "A股数据转换（含过滤）Parquet -> Qlib Bin"),
    "convert-hk":    (convert_hk,    "港股数据转换 Parquet -> Qlib Bin"),
    "train":         (train,         "训练全 A 股 LightGBM 模型"),
    "train-hk":      (train_hk,      "训练港股选股模型"),
    "backtest":      (backtest,      "A股 TopkDropout 回测"),
    "backtest-hk":   (backtest_hk,   "港股 TopkDropout 回测"),
    "train-index":   (train_index,   "训练恒指/科技指数预测模型"),
    "predict-index": (predict_index, "预测恒指/科技指数明日走势"),
    "dryrun":        (dryrun,        "执行模拟交易 (TopkDropout Dry Run)"),
    "dryrun-status": (dryrun_status, "查看模拟交易组合当前状态"),
    "deploy":        (deploy,        "部署 pred.pkl 到 trader"),
    "dashboard":     (dashboard,     "启动 Streamlit 看板"),
}


def main() -> None:
    if len(sys.argv) <= 1 or sys.argv[1] in ("-h", "--help"):
        print("用法: python main.py <命令>\n")
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
