"""
指数时序预测：恒生指数(HK.800000) / 恒生科技(HK.800700)
基于技术指标特征 + LightGBM，预测次日涨跌方向与幅度。
不依赖 Qlib，直接读取 Parquet 数据。
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd


# ── 数据路径 ──────────────────────────────────────────────────────────────────

def _data_dir() -> Path:
    root = Path(__file__).resolve().parent.parent
    for sub in ["data/kline/K_DAY", "data/kline/1d"]:
        p = root / sub
        if p.exists():
            return p
    raise FileNotFoundError(f"找不到 K_DAY 数据目录，请检查 {root}/data/")


def _models_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "models"


def load_index_data(code: str) -> pd.DataFrame:
    """读取指数 Parquet，返回按日期排序的 DataFrame。"""
    base = _data_dir()
    path = base / code / "data.parquet"
    if not path.exists():
        raise FileNotFoundError(f"找不到 {code} 数据: {path}")
    df = pd.read_parquet(path)
    # 统一日期列
    date_col = "time_key" if "time_key" in df.columns else "date"
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.rename(columns={date_col: "date"})
    df = df.sort_values("date").reset_index(drop=True)
    # amount 列
    if "amount" not in df.columns:
        for alias in ["turnover", "turn_over"]:
            if alias in df.columns:
                df["amount"] = df[alias]
                break
    return df


def _load_cross_asset(code: str) -> pd.DataFrame | None:
    """
    尝试读取跨资产代码（如 US.SPY、US.QQQ、HK.800000）的日线数据。
    若文件不存在则返回 None（graceful fallback）。
    """
    try:
        base = _data_dir()
        path = base / code / "data.parquet"
        if not path.exists():
            return None
        df = pd.read_parquet(path)
        date_col = "time_key" if "time_key" in df.columns else "date"
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.rename(columns={date_col: "date"})
        df = df.sort_values("date").set_index("date")
        return df[["close"]].rename(columns={"close": code})
    except Exception:
        return None


def _merge_cross_asset_features(d: pd.DataFrame) -> pd.DataFrame:
    """
    将跨资产收益率特征 merge 到主 DataFrame 中。
    按日期左连接，若目标数据不存在则对应列置 0。

    跨资产代码:
      US.SPY  - 标普500 ETF (全球风险偏好)
      US.QQQ  - 纳斯达克 ETF (科技股风险偏好)
    """
    d = d.copy().set_index("date")

    cross_assets = {
        "spy": "US.SPY",
        "qqq": "US.QQQ",
    }

    for col_prefix, asset_code in cross_assets.items():
        asset_df = _load_cross_asset(asset_code)
        if asset_df is not None:
            # 注意：US市场收盘时间比港股早（UTC-5），HK开盘时港股可见昨日美股收益
            # 所以用美股 t 日收益预测港股 t+1 日，对齐到 close 日期
            asset_close = asset_df[asset_code].rename(f"_{col_prefix}_c")
            d = d.join(asset_close, how="left")
            d[f"{col_prefix}_ret1"] = d[f"_{col_prefix}_c"].pct_change(1)
            d[f"{col_prefix}_ret5"] = d[f"_{col_prefix}_c"].pct_change(5)
            d = d.drop(columns=[f"_{col_prefix}_c"])
        else:
            d[f"{col_prefix}_ret1"] = 0.0
            d[f"{col_prefix}_ret5"] = 0.0

    return d.reset_index()


# ── 特征工程 ──────────────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    从 OHLCV 构建时序技术特征 + 跨资产特征，目标列：next_ret（次日收益率）。
    跨资产特征（SPY/QQQ）若数据不存在则填 0。
    """
    d = _merge_cross_asset_features(df)
    c = d["close"].astype(float)
    o = d["open"].astype(float)
    h = d["high"].astype(float)
    l = d["low"].astype(float)
    v = d["volume"].astype(float).replace(0, np.nan)

    # ── 收益率特征 ──
    for n in [1, 2, 3, 5, 10, 20]:
        d[f"ret_{n}d"] = c.pct_change(n)

    # ── 均线与价格位置 ──
    for n in [5, 10, 20, 60]:
        ma = c.rolling(n).mean()
        d[f"ma{n}_ratio"] = c / ma - 1

    # ── 波动率 ──
    ret1 = c.pct_change(1)
    for n in [5, 10, 20]:
        d[f"vol_{n}d"] = ret1.rolling(n).std()

    # ── RSI(14) ──
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    d["rsi14"] = 100 - 100 / (1 + gain / (loss + 1e-9))

    # ── MACD ──
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    d["macd"] = macd / (c + 1e-9)
    d["macd_hist"] = (macd - signal) / (c + 1e-9)

    # ── Bollinger Band 位置 ──
    ma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    d["bb_pos"] = (c - ma20) / (2 * std20 + 1e-9)

    # ── ATR 归一化 ──
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    d["atr14"] = tr.rolling(14).mean() / (c + 1e-9)

    # ── 成交量特征 ──
    # 成交量特征（指数 volume 可能为 0，改用 amount 或置 0）
    if v.isna().all() or (v == 0).all():
        amt = d.get("amount", pd.Series(dtype=float))
        if isinstance(amt, pd.Series) and not amt.isna().all() and not (amt == 0).all():
            v = amt.astype(float).replace(0, np.nan)
        else:
            d["vol_ratio5"]  = 0.0
            d["vol_ratio20"] = 0.0
            v = None
    if v is not None:
        d["vol_ratio5"]  = v / v.rolling(5).mean()
        d["vol_ratio20"] = v / v.rolling(20).mean()

    # ── 高低点突破 ──
    d["high20_break"] = (c > h.rolling(20).max().shift(1)).astype(float)
    d["low20_break"]  = (c < l.rolling(20).min().shift(1)).astype(float)

    # ── 目标：次日收益率（回归）和方向（分类）──
    d["next_ret"] = c.shift(-1) / c - 1
    d["next_dir"] = (d["next_ret"] > 0).astype(int)

    # 只删除核心特征有 NaN 的行（保留成交量特征为 0 的行）
    core_cols = [c for c in FEATURE_COLS if not c.startswith("vol_ratio")]
    d[FEATURE_COLS] = d[FEATURE_COLS].ffill().fillna(0)
    d = d.dropna(subset=core_cols + ["next_ret"])
    return d


FEATURE_COLS = [
    "ret_1d", "ret_2d", "ret_3d", "ret_5d", "ret_10d", "ret_20d",
    "ma5_ratio", "ma10_ratio", "ma20_ratio", "ma60_ratio",
    "vol_5d", "vol_10d", "vol_20d",
    "rsi14", "macd", "macd_hist", "bb_pos", "atr14",
    "vol_ratio5", "vol_ratio20",
    "high20_break", "low20_break",
    # 跨资产特征（若数据不存在则为 0）
    "spy_ret1", "spy_ret5",
    "qqq_ret1", "qqq_ret5",
]


# ── 训练 ──────────────────────────────────────────────────────────────────────

def train_index_model(
    code: str = "HK.800000",
    test_ratio: float = 0.2,
    verbose: bool = True,
) -> dict:
    """
    训练单只指数的次日预测模型（LightGBM 回归）。

    Args:
        code:       指数代码，如 HK.800000（恒指）或 HK.800700（恒生科技）
        test_ratio: 测试集比例（按时间顺序末尾截取）
        verbose:    是否打印结果

    Returns:
        dict with keys: code, test_ic, test_dir_acc, model_path
    """
    import lightgbm as lgb
    from sklearn.metrics import accuracy_score
    from scipy.stats import pearsonr

    df = load_index_data(code)
    df = build_features(df)

    n = len(df)
    split = int(n * (1 - test_ratio))
    train_df = df.iloc[:split]
    test_df  = df.iloc[split:]

    X_train = train_df[FEATURE_COLS].values
    y_train = train_df["next_ret"].values
    X_test  = test_df[FEATURE_COLS].values
    y_test  = test_df["next_ret"].values

    if verbose:
        print(f"\n=== 指数预测模型训练：{code} ===")
        print(f"  训练集: {train_df['date'].iloc[0].date()} ~ {train_df['date'].iloc[-1].date()} ({len(train_df)} 天)")
        print(f"  测试集: {test_df['date'].iloc[0].date()} ~ {test_df['date'].iloc[-1].date()} ({len(test_df)} 天)")

    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "lambda_l1": 1.0,
        "lambda_l2": 1.0,
        "n_estimators": 500,
        "n_jobs": -1,
        "verbose": -1,
    }

    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=0)],
    )

    pred_test = model.predict(X_test)
    ic, _ = pearsonr(pred_test, y_test)
    dir_acc = accuracy_score((y_test > 0).astype(int), (pred_test > 0).astype(int))

    if verbose:
        print(f"  测试集 IC（相关系数）: {ic:.4f}")
        print(f"  方向准确率:           {dir_acc:.2%}")
        feat_imp = pd.Series(model.feature_importances_, index=FEATURE_COLS)
        print(f"  Top-5 重要特征: {feat_imp.nlargest(5).index.tolist()}")

    # 保存模型
    models_dir = _models_dir()
    models_dir.mkdir(exist_ok=True)
    safe_code = code.replace(".", "_")
    model_path = models_dir / f"index_{safe_code}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"model": model, "code": code, "features": FEATURE_COLS}, f)

    if verbose:
        print(f"  模型已保存: {model_path}")

    return {"code": code, "test_ic": ic, "test_dir_acc": dir_acc, "model_path": str(model_path)}


# ── 预测 ──────────────────────────────────────────────────────────────────────

def predict_index_tomorrow(code: str = "HK.800000") -> dict:
    """
    加载已训练模型，预测该指数明日涨跌。

    Returns:
        dict: date, pred_ret (预测收益率), direction (up/down), confidence
    """
    safe_code = code.replace(".", "_")
    model_path = _models_dir() / f"index_{safe_code}.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"未找到模型: {model_path}，请先调用 train_index_model('{code}')")

    with open(model_path, "rb") as f:
        obj = pickle.load(f)
    model = obj["model"]
    features = obj["features"]

    df = load_index_data(code)
    df = build_features(df)
    # 用最后一行（最近交易日）预测
    last = df.iloc[[-1]]
    X = last[features].values
    pred_ret = float(model.predict(X)[0])

    return {
        "code": code,
        "last_date": str(last["date"].iloc[0].date()),
        "pred_next_ret": pred_ret,
        "direction": "up" if pred_ret > 0 else "down",
        "last_close": float(last["close"].iloc[0]),
        "pred_close_est": float(last["close"].iloc[0]) * (1 + pred_ret),
    }


# ── 批量训练 & 预测 ──────────────────────────────────────────────────────────

INDEX_CODES = ["HK.800000", "HK.800700"]


def train_all_index_models(verbose: bool = True) -> list[dict]:
    """训练所有指数模型（恒指 + 恒生科技）。"""
    results = []
    for code in INDEX_CODES:
        try:
            r = train_index_model(code, verbose=verbose)
            results.append(r)
        except Exception as e:
            print(f"[ERROR] {code} 训练失败: {e}")
    return results


def predict_all_indexes() -> list[dict]:
    """预测所有指数明日走势。"""
    results = []
    for code in INDEX_CODES:
        try:
            r = predict_index_tomorrow(code)
            print(f"{code}: {r['direction'].upper():4s} | 预测收益 {r['pred_next_ret']:+.2%} | "
                  f"末日 {r['last_date']} 收 {r['last_close']:.1f} → 预估 {r['pred_close_est']:.1f}")
            results.append(r)
        except Exception as e:
            print(f"[ERROR] {code}: {e}")
    return results
