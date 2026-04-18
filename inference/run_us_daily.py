from __future__ import annotations

import json
import logging
import math
import os
import re
import signal
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from futu import OpenQuoteContext, OpenSecTradeContext, TrdMarket, RET_OK, SysConfig

from collector.futu_client import FutuClient
from trader.trade_daily import (
    FUTU_HOST,
    FUTU_PORT,
    FUTU_SIM_ACC_ID,
    _coerce_float,
)
from trader import trade_daily as trade_base

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("us_daily_inference")

DATA_DIR = Path(os.environ.get("DATA_DIR", str(Path.home() / "quantpilot_data")))
SIGNAL_DIR = Path(os.environ.get("SIGNAL_DIR", str(DATA_DIR / "signals")))
US_SIGNAL_DIR = Path(os.environ.get("US_SIGNAL_DIR", str(SIGNAL_DIR / "us")))
US_SIGNAL_OUTPUT_TAG = os.environ.get("US_SIGNAL_OUTPUT_TAG", "").strip()
US_UNIVERSE = os.environ.get("US_UNIVERSE", "sp500").strip().lower()
US_TARGET_CODES = os.environ.get("US_TARGET_CODES", "").strip()
US_TARGET_INDEXES = os.environ.get("US_TARGET_INDEXES", "").strip()
US_UNIVERSE_FILE = os.environ.get("US_UNIVERSE_FILE", "").strip()
US_MIN_PRICE = float(os.environ.get("US_MIN_PRICE", "5"))
US_MIN_DOLLAR_VOLUME = float(os.environ.get("US_MIN_DOLLAR_VOLUME", "10000000"))
US_ANALYSIS_TOP_K = int(os.environ.get("US_ANALYSIS_TOP_K", "20"))
US_MAX_POSITIONS = int(os.environ.get("US_MAX_POSITIONS", "5"))
US_ANALYSIS_TIMEOUT_SECONDS = int(os.environ.get("US_ANALYSIS_TIMEOUT_SECONDS", "3600"))
US_ANALYSIS_CONCURRENCY = int(os.environ.get("US_ANALYSIS_CONCURRENCY", "10"))
FUTU_CONNECT_TIMEOUT_SECONDS = int(os.environ.get("US_FUTU_CONNECT_TIMEOUT_SECONDS", "30"))
FUTU_RSA_KEY = os.environ.get("FUTU_RSA_KEY", "")
TRADING_AGENTS_DIR = Path(
    os.environ.get(
        "TRADING_AGENTS_DIR",
        str(Path.home() / ".openclaw" / "workspace" / "trading-agents"),
    )
)
TRADING_AGENTS_SCRIPT = Path(
    os.environ.get("TRADING_AGENTS_SCRIPT", str(TRADING_AGENTS_DIR / "run_analysis.sh"))
)
SP500_WIKI_URL = os.environ.get(
    "SP500_WIKI_URL",
    "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
)
ACTION_RE = re.compile(r"最终交易提案\s*[:：]\s*(BUY|HOLD|SELL)\b", re.IGNORECASE)
RATING_PATTERNS = [
    re.compile(r"评级\s*[:：]\s*(Buy|Overweight|Hold|Underweight|Sell)\b", re.IGNORECASE),
    re.compile(r"\*\*评级\*\*\s*[:：]?\s*(Buy|Overweight|Hold|Underweight|Sell)\b", re.IGNORECASE),
]
RATING_SCORES = {
    "SELL": -2,
    "UNDERWEIGHT": -1,
    "HOLD": 0,
    "OVERWEIGHT": 1,
    "BUY": 2,
}


def _output_tag() -> str:
    return US_SIGNAL_OUTPUT_TAG or datetime.now().strftime("%Y%m%d")


def _replace_symlink(link: Path, target: Path) -> None:
    tmp_link = link.with_name(f".{link.name}.tmp")
    tmp_link.unlink(missing_ok=True)
    tmp_link.symlink_to(target.name)
    tmp_link.replace(link)


def normalize_us_code(symbol: str) -> str:
    raw = str(symbol or "").strip().upper()
    if not raw:
        raise ValueError("empty symbol")
    if raw.startswith("US."):
        raw = raw[3:]
    raw = raw.replace("-", ".")
    return f"US.{raw}"


def _snapshot_value(snapshot: dict[str, object], *keys: str) -> float | None:
    for key in keys:
        value = snapshot.get(key)
        num = _coerce_float(value)
        if num is not None:
            return num
    return None


def _parse_code_list(raw: str) -> list[str]:
    values = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            values.append(normalize_us_code(item))
    return sorted(set(values))


def _load_codes_from_file(path: Path) -> list[str]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            values = payload
        elif isinstance(payload, dict):
            values = payload.get("codes", [])
        else:
            raise ValueError(f"unsupported universe json format: {type(payload)}")
        return sorted({normalize_us_code(value) for value in values})

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        for column in ("code", "symbol", "ticker"):
            if column in df.columns:
                return sorted({normalize_us_code(value) for value in df[column].dropna().tolist()})
        raise ValueError(f"csv missing code/symbol/ticker column: {path}")

    values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return sorted({normalize_us_code(value) for value in values})


def fetch_sp500_codes() -> list[str]:
    tables = pd.read_html(SP500_WIKI_URL)
    for table in tables:
        if "Symbol" in table.columns:
            symbols = [normalize_us_code(value) for value in table["Symbol"].dropna().tolist()]
            codes = sorted(set(symbols))
            if codes:
                return codes
    raise RuntimeError("failed to locate Symbol column in S&P 500 table")


def fetch_futu_index_codes(index_codes: list[str]) -> list[str]:
    client = FutuClient(FUTU_HOST, FUTU_PORT)
    if not client.connect():
        raise RuntimeError(f"failed to connect Futu OpenD {FUTU_HOST}:{FUTU_PORT}")
    try:
        codes: list[str] = []
        for index_code in index_codes:
            codes.extend(client.get_index_constituents(index_code))
        return sorted({normalize_us_code(code) for code in codes if str(code).startswith("US.")})
    finally:
        client.disconnect()


def load_universe_codes() -> tuple[list[str], str]:
    if US_TARGET_CODES:
        return _parse_code_list(US_TARGET_CODES), "env_target_codes"

    if US_UNIVERSE_FILE:
        path = Path(US_UNIVERSE_FILE).expanduser()
        return _load_codes_from_file(path), f"file:{path}"

    if US_TARGET_INDEXES:
        index_codes = [item.strip() for item in US_TARGET_INDEXES.split(",") if item.strip()]
        return fetch_futu_index_codes(index_codes), f"futu_indexes:{','.join(index_codes)}"

    if US_UNIVERSE == "sp500":
        return fetch_sp500_codes(), "sp500_wikipedia"

    raise RuntimeError(
        "US universe not configured; set US_TARGET_CODES, US_UNIVERSE_FILE, US_TARGET_INDEXES, or US_UNIVERSE=sp500"
    )


def _open_us_contexts() -> tuple[OpenSecTradeContext, OpenQuoteContext]:
    def _timeout_handler(signum, frame):
        raise TimeoutError(f"OpenD 连接超时 ({FUTU_HOST}:{FUTU_PORT})")

    if FUTU_RSA_KEY and Path(FUTU_RSA_KEY).is_file():
        SysConfig.enable_proto_encrypt(True)
        SysConfig.set_init_rsa_file(FUTU_RSA_KEY)

    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(FUTU_CONNECT_TIMEOUT_SECONDS)
    try:
        trd_ctx = OpenSecTradeContext(filter_trdmarket=TrdMarket.US, host=FUTU_HOST, port=FUTU_PORT)
        quote_ctx = OpenQuoteContext(host=FUTU_HOST, port=FUTU_PORT)
        return trd_ctx, quote_ctx
    finally:
        signal.alarm(0)


def get_us_positions() -> dict[str, dict[str, Any]]:
    trd_ctx, quote_ctx = _open_us_contexts()
    quote_ctx.close()
    try:
        ret, acc_list = trd_ctx.get_acc_list()
        if ret != RET_OK:
            raise RuntimeError(f"get_acc_list failed: {acc_list}")
        acc_id = trade_base.select_sim_acc_id(acc_list, preferred_acc_id=FUTU_SIM_ACC_ID)
        return trade_base.get_positions(trd_ctx, acc_id=acc_id, refresh_cache=True)
    finally:
        trd_ctx.close()


def get_us_snapshots(codes: list[str]) -> dict[str, dict[str, object]]:
    trd_ctx, quote_ctx = _open_us_contexts()
    trd_ctx.close()
    try:
        return trade_base.get_market_snapshots(quote_ctx, codes)
    finally:
        quote_ctx.close()


def build_candidate_frame(codes: list[str], snapshots: dict[str, dict[str, object]]) -> pd.DataFrame:
    rows = []
    for code in codes:
        snapshot = snapshots.get(code)
        if not snapshot:
            continue
        price = _snapshot_value(snapshot, "last_price", "nominal_price", "prev_close_price")
        turnover = _snapshot_value(snapshot, "turnover") or 0.0
        volume = _snapshot_value(snapshot, "volume") or 0.0
        change_rate = float(snapshot.get("change_rate") or 0.0)
        if price is None or price < US_MIN_PRICE:
            continue
        dollar_volume = turnover or price * volume
        if dollar_volume < US_MIN_DOLLAR_VOLUME:
            continue
        score = change_rate + math.log10(max(dollar_volume, 1.0))
        rows.append(
            {
                "code": code,
                "price": round(price, 4),
                "change_rate": round(change_rate, 4),
                "dollar_volume": round(float(dollar_volume), 2),
                "candidate_score": round(score, 6),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["code", "price", "change_rate", "dollar_volume", "candidate_score"])
    return pd.DataFrame(rows).sort_values(
        ["candidate_score", "dollar_volume", "code"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def _state_path_for_ticker(ticker: str) -> Path:
    today = datetime.now().strftime("%Y-%m-%d")
    return TRADING_AGENTS_DIR / "results" / ticker / f"{today}_state.json"


def parse_trade_action(state: dict[str, Any]) -> str | None:
    text = str(state.get("trade_proposal", ""))
    match = ACTION_RE.search(text)
    if match:
        return match.group(1).upper()
    return None


def parse_rating(state: dict[str, Any]) -> str | None:
    text = str(state.get("final_decision", ""))
    for pattern in RATING_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1).upper()
    head = text[:300]
    for label in ("OVERWEIGHT", "UNDERWEIGHT", "BUY", "HOLD", "SELL"):
        if re.search(rf"\b{label}\b", head, re.IGNORECASE):
            return label
    return None


def run_deep_analysis(code: str) -> dict[str, Any]:
    ticker = code.split(".", 1)[1]
    if not TRADING_AGENTS_SCRIPT.exists():
        raise FileNotFoundError(f"trading agents script missing: {TRADING_AGENTS_SCRIPT}")
    result = subprocess.run(
        [str(TRADING_AGENTS_SCRIPT), ticker],
        cwd=str(TRADING_AGENTS_DIR),
        capture_output=True,
        text=True,
        timeout=US_ANALYSIS_TIMEOUT_SECONDS,
        check=False,
    )
    if result.returncode != 0:
        stderr_tail = "\n".join(result.stderr.splitlines()[-20:])
        stdout_tail = "\n".join(result.stdout.splitlines()[-20:])
        raise RuntimeError(
            f"deep-analysis failed for {code}: rc={result.returncode}\nSTDERR:\n{stderr_tail}\nSTDOUT:\n{stdout_tail}"
        )
    state_path = _state_path_for_ticker(ticker)
    if not state_path.exists():
        raise FileNotFoundError(f"deep-analysis state not found: {state_path}")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["state_path"] = str(state_path)
    return state


def _analyze_code(code: str, candidate_scores: dict[str, float]) -> dict[str, Any]:
    log.info(f"Analyzing {code} via deep-analysis ...")
    state = run_deep_analysis(code)
    action = parse_trade_action(state)
    rating = parse_rating(state)
    return {
        "code": code,
        "action": action or "HOLD",
        "rating": rating or "HOLD",
        "decision_score": RATING_SCORES.get((rating or "HOLD").upper(), 0),
        "candidate_score": float(candidate_scores.get(code, 0.0)),
        "run_id": state.get("run_id"),
        "state_path": state.get("state_path"),
        "trade_proposal": state.get("trade_proposal", ""),
        "final_decision": state.get("final_decision", ""),
        "investment_plan": state.get("investment_plan", ""),
    }


def analyze_codes(analysis_codes: list[str], candidate_scores: dict[str, float]) -> list[dict[str, Any]]:
    if not analysis_codes:
        return []

    max_workers = max(1, min(US_ANALYSIS_CONCURRENCY, len(analysis_codes)))
    results_by_code: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(_analyze_code, code, candidate_scores): code
            for code in analysis_codes
        }
        for future in as_completed(future_map):
            code = future_map[future]
            try:
                results_by_code[code] = future.result()
            except Exception as exc:
                log.error(f"deep-analysis failed for {code}: {exc}")

    return [results_by_code[code] for code in analysis_codes if code in results_by_code]


def build_trade_plan(
    analyses: list[dict[str, Any]],
    current_positions: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    current_codes = set(current_positions.keys())
    current_holds = []
    buy_candidates = []
    analysis_by_code = {item["code"]: item for item in analyses}

    for item in analyses:
        code = item["code"]
        action = item["action"].upper()
        if action == "BUY":
            buy_candidates.append(item)
        elif action == "HOLD" and code in current_codes:
            current_holds.append(item)

    current_holds.sort(
        key=lambda item: (item["decision_score"], item["candidate_score"], item["code"]),
        reverse=True,
    )
    buy_candidates.sort(
        key=lambda item: (item["decision_score"], item["candidate_score"], item["code"]),
        reverse=True,
    )
    selected_holds = current_holds[:US_MAX_POSITIONS]
    remaining_slots = max(US_MAX_POSITIONS - len(selected_holds), 0)
    selected_buys = buy_candidates[:remaining_slots]
    target_codes = {item["code"] for item in selected_holds} | {item["code"] for item in selected_buys}

    orders = []
    target_count = max(len(target_codes), 1)
    target_weight = round(1.0 / target_count, 6)

    for code in sorted(current_codes | target_codes):
        analysis = analysis_by_code.get(code)
        if code in target_codes and code in current_codes:
            action = "HOLD"
            weight = target_weight
        elif code in target_codes:
            action = "BUY"
            weight = target_weight
        else:
            action = "SELL"
            weight = 0.0
        orders.append(
            {
                "code": code,
                "action": action,
                "target_weight": weight,
                "reason": (analysis or {}).get("trade_proposal", "")[:400],
                "rating": (analysis or {}).get("rating", "N/A"),
                "run_id": (analysis or {}).get("run_id"),
                "state_path": (analysis or {}).get("state_path"),
            }
        )

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "max_positions": US_MAX_POSITIONS,
        "target_codes": sorted(target_codes),
        "current_codes": sorted(current_codes),
        "orders": orders,
    }


def write_outputs(payload: dict[str, Any]) -> dict[str, Path]:
    US_SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    tag = _output_tag()
    paths: dict[str, Path] = {}

    output_map = {
        "candidates": US_SIGNAL_DIR / f"us_candidates_{tag}.json",
        "analyses": US_SIGNAL_DIR / f"us_analyses_{tag}.json",
        "plan": US_SIGNAL_DIR / f"us_trade_plan_{tag}.json",
    }
    for key, path in output_map.items():
        tmp_path = path.with_suffix(f"{path.suffix}.tmp")
        tmp_path.write_text(json.dumps(payload[key], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp_path.replace(path)
        paths[key] = path

    latest_map = {
        "latest_candidates": (US_SIGNAL_DIR / "us_candidates_latest.json", paths["candidates"]),
        "latest_analyses": (US_SIGNAL_DIR / "us_analyses_latest.json", paths["analyses"]),
        "latest_plan": (US_SIGNAL_DIR / "us_trade_plan_latest.json", paths["plan"]),
    }
    for key, (link, target) in latest_map.items():
        _replace_symlink(link, target)
        paths[key] = link

    return paths


def run_us_daily() -> dict[str, Any]:
    universe_codes, universe_source = load_universe_codes()
    log.info(f"US universe loaded: source={universe_source} count={len(universe_codes)}")
    current_positions = get_us_positions()
    current_codes = sorted(current_positions.keys())
    log.info(f"Current US positions: {current_codes or 'empty'}")

    snapshot_codes = sorted(set(universe_codes) | set(current_codes))
    snapshots = get_us_snapshots(snapshot_codes)
    candidate_df = build_candidate_frame(universe_codes, snapshots)
    if candidate_df.empty and not current_codes:
        raise RuntimeError("no eligible US candidates after price/liquidity filter")

    candidate_scores = {
        row["code"]: float(row["candidate_score"])
        for row in candidate_df.to_dict(orient="records")
    }
    top_candidates = candidate_df.head(US_ANALYSIS_TOP_K)["code"].tolist()
    analysis_codes = sorted(set(top_candidates) | set(current_codes))
    analyses = analyze_codes(analysis_codes, candidate_scores)
    if not analyses:
        raise RuntimeError("deep-analysis produced no usable US decisions")

    analyzed_codes = {item["code"] for item in analyses}
    missing_held = sorted(set(current_codes) - analyzed_codes)
    if missing_held:
        raise RuntimeError(
            f"missing deep-analysis results for held positions: {', '.join(missing_held)}"
        )

    plan = build_trade_plan(analyses, current_positions)
    payload = {
        "candidates": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "source": universe_source,
            "universe_count": len(universe_codes),
            "analysis_codes": analysis_codes,
            "rows": candidate_df.head(max(US_ANALYSIS_TOP_K * 3, US_ANALYSIS_TOP_K)).to_dict(orient="records"),
        },
        "analyses": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "rows": analyses,
        },
        "plan": plan,
    }
    paths = write_outputs(payload)
    return {
        "universe_source": universe_source,
        "universe_count": len(universe_codes),
        "candidate_count": len(candidate_df),
        "analysis_count": len(analyses),
        "paths": {key: str(value) for key, value in paths.items()},
    }


def main() -> None:
    log.info("=" * 50)
    log.info("QuantPilot US daily pipeline")
    log.info("=" * 50)
    try:
        result = run_us_daily()
        log.info(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as exc:
        log.error(f"US daily pipeline failed: {exc}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
