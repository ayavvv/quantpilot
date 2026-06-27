"""Build the actual US microstructure collection universe.

The coarse universe is the same-day discovery pool.  This script adds a small
rolling follow-up layer so strong prior-day signals keep receiving tick/order
book/quote coverage for a few days without exploding the collector to every
historical candidate.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable
from zoneinfo import ZoneInfo

import pandas as pd

from scripts.build_us_microstructure_universe import DEFAULT_BASE_DIR, DEFAULT_NAS_DIR
from scripts.collect_us_microstructure import _sync_paths_to_nas
from strategy.us_microstructure_features import normalize_us_symbol, normalize_us_symbols


US_EASTERN = ZoneInfo("America/New_York")
STATUS_SCHEMA_VERSION = 1
DEFAULT_FOLLOWUP_DAYS = 2
DEFAULT_FOLLOWUP_MAX_SYMBOLS = 100
DEFAULT_FOLLOWUP_MIN_SCORE = 55.0
DEFAULT_MAX_TOTAL_SYMBOLS = 124
DEFAULT_FOLLOWUP_CONFIDENCE = "high,watch"
DEFAULT_FOLLOWUP_STAGE_KEYWORDS = "watch"


def _collection_date_from_utc(value: datetime | None = None) -> str:
    timestamp = value or datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(US_EASTERN).strftime("%Y-%m-%d")


def _split_csv(value: object) -> set[str]:
    return {item.strip().lower() for item in str(value or "").split(",") if item.strip()}


def _read_symbol_file(path: str | Path | None) -> list[str]:
    if not path:
        return []
    resolved = Path(path).expanduser()
    if not resolved.exists():
        return []
    return normalize_us_symbols(line.strip() for line in resolved.read_text(encoding="utf-8").splitlines())


def _number(value: object, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if pd.isna(parsed):
        return default
    return float(parsed)


def _bool_value(value: object, default: bool = True) -> bool:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    text = str(value).strip().lower()
    if not text:
        return default
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    return default


def _signal_score(row: pd.Series) -> float:
    for column in ("side_score", "score"):
        if column in row:
            value = _number(row.get(column), default=float("nan"))
            if pd.notna(value):
                return value
    return max(_number(row.get("accumulation_score")), _number(row.get("distribution_score")))


def _confidence_priority(value: object) -> int:
    text = str(value or "").strip().lower()
    return {"high": 3, "watch": 2, "diagnostic": 1, "warmup": 0}.get(text, 0)


def _rank_value(value: object) -> int:
    parsed = _number(value, default=999999.0)
    return int(parsed) if parsed > 0 else 999999


def _signal_rows_for_date(
    base_dir: Path,
    date_value: str,
    *,
    min_score: float,
    include_confidence: set[str],
    stage_keywords: set[str],
    require_data_quality: bool,
) -> list[dict[str, object]]:
    path = base_dir / "signals" / f"date={date_value}" / "us_major_flow_signals.csv"
    if not path.exists():
        return []
    try:
        frame = pd.read_csv(path)
    except Exception:
        return []
    if frame.empty or "symbol" not in frame.columns:
        return []

    rows: list[dict[str, object]] = []
    for _, row in frame.iterrows():
        symbol = normalize_us_symbol(row.get("symbol"))
        if not symbol:
            continue
        if require_data_quality and not _bool_value(row.get("data_quality_pass"), default=True):
            continue
        confidence = str(row.get("confidence") or "").strip().lower()
        stage = str(row.get("stage") or "").strip().lower()
        score = _signal_score(row)
        confidence_pass = confidence in include_confidence if include_confidence else False
        stage_pass = any(keyword in stage for keyword in stage_keywords) if stage_keywords else False
        score_pass = score >= float(min_score)
        if not (confidence_pass or stage_pass or score_pass):
            continue
        rows.append(
            {
                "symbol": symbol,
                "collection_source": "followup",
                "origin_date": date_value,
                "origin_rank": _rank_value(row.get("rank")),
                "origin_score": round(float(score), 6),
                "origin_stage": str(row.get("stage") or ""),
                "origin_confidence": str(row.get("confidence") or ""),
                "origin_side": str(row.get("side") or ""),
                "origin_reason": str(row.get("reason") or ""),
                "priority_score": (
                    _confidence_priority(confidence) * 1000.0
                    + (100.0 if stage_pass else 0.0)
                    + float(score)
                ),
            }
        )
    return rows


def _collect_followup_rows(
    base_dir: Path,
    *,
    date_value: str,
    lookback_days: int,
    min_score: float,
    include_confidence: set[str],
    stage_keywords: set[str],
    require_data_quality: bool,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    current_date = datetime.strptime(date_value[:10], "%Y-%m-%d").date()
    for offset in range(1, max(0, int(lookback_days)) + 1):
        origin = (current_date - timedelta(days=offset)).isoformat()
        for row in _signal_rows_for_date(
            base_dir,
            origin,
            min_score=min_score,
            include_confidence=include_confidence,
            stage_keywords=stage_keywords,
            require_data_quality=require_data_quality,
        ):
            row["age_days"] = offset
            row["expires_date"] = (datetime.strptime(str(row["origin_date"])[:10], "%Y-%m-%d").date() + timedelta(days=lookback_days)).isoformat()
            rows.append(row)
    return rows


def build_collection_universe(
    *,
    base_dir: str | Path,
    date_value: str,
    current_universe_file: str | Path,
    followup_days: int = DEFAULT_FOLLOWUP_DAYS,
    followup_max_symbols: int = DEFAULT_FOLLOWUP_MAX_SYMBOLS,
    followup_min_score: float = DEFAULT_FOLLOWUP_MIN_SCORE,
    max_total_symbols: int = DEFAULT_MAX_TOTAL_SYMBOLS,
    followup_confidence: object = DEFAULT_FOLLOWUP_CONFIDENCE,
    followup_stage_keywords: object = DEFAULT_FOLLOWUP_STAGE_KEYWORDS,
    require_data_quality: bool = True,
) -> tuple[pd.DataFrame, dict[str, object]]:
    base = Path(base_dir).expanduser()
    current_symbols = _read_symbol_file(current_universe_file)
    include_confidence = _split_csv(followup_confidence)
    stage_keywords = _split_csv(followup_stage_keywords)

    current_rows = [
        {
            "symbol": symbol,
            "collection_source": "current",
            "origin_date": date_value[:10],
            "age_days": 0,
            "expires_date": date_value[:10],
            "origin_rank": index,
            "origin_score": "",
            "origin_stage": "",
            "origin_confidence": "",
            "origin_side": "",
            "origin_reason": "",
            "priority_score": 10000.0 - index,
        }
        for index, symbol in enumerate(current_symbols, start=1)
    ]
    current_set = set(current_symbols)

    followup_candidates = _collect_followup_rows(
        base,
        date_value=date_value[:10],
        lookback_days=followup_days,
        min_score=followup_min_score,
        include_confidence=include_confidence,
        stage_keywords=stage_keywords,
        require_data_quality=require_data_quality,
    )
    followup_by_symbol: dict[str, dict[str, object]] = {}
    for row in followup_candidates:
        symbol = str(row.get("symbol") or "")
        if not symbol or symbol in current_set:
            continue
        existing = followup_by_symbol.get(symbol)
        if existing is None:
            followup_by_symbol[symbol] = row
            continue
        key = (float(row.get("priority_score") or 0.0), -int(row.get("age_days") or 999), -_rank_value(row.get("origin_rank")))
        existing_key = (
            float(existing.get("priority_score") or 0.0),
            -int(existing.get("age_days") or 999),
            -_rank_value(existing.get("origin_rank")),
        )
        if key > existing_key:
            followup_by_symbol[symbol] = row

    followup_rows = sorted(
        followup_by_symbol.values(),
        key=lambda row: (
            -float(row.get("priority_score") or 0.0),
            int(row.get("age_days") or 999),
            _rank_value(row.get("origin_rank")),
            str(row.get("symbol") or ""),
        ),
    )
    followup_limit = max(0, int(followup_max_symbols))
    if max_total_symbols > 0:
        followup_limit = min(followup_limit, max(0, int(max_total_symbols) - len(current_rows)))
    followup_rows = followup_rows[:followup_limit]

    rows = current_rows + followup_rows
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.drop_duplicates("symbol", keep="first").reset_index(drop=True)
        frame["collection_rank"] = range(1, len(frame) + 1)
        keep = [
            "collection_rank",
            "symbol",
            "collection_source",
            "origin_date",
            "age_days",
            "expires_date",
            "origin_rank",
            "origin_score",
            "origin_stage",
            "origin_confidence",
            "origin_side",
            "origin_reason",
            "priority_score",
        ]
        frame = frame[[column for column in keep if column in frame.columns]]

    source_counts = frame["collection_source"].value_counts().to_dict() if not frame.empty else {}
    origin_counts = frame[frame["collection_source"] == "followup"]["origin_date"].value_counts().sort_index().to_dict() if not frame.empty else {}
    status = {
        "status_schema_version": STATUS_SCHEMA_VERSION,
        "status": "ok" if not frame.empty else "empty",
        "date": date_value[:10],
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "base_dir": str(base),
        "layer": "collection_universe",
        "current_universe_file": str(Path(current_universe_file).expanduser()),
        "current_symbol_count": int(len(current_symbols)),
        "followup_days": int(followup_days),
        "followup_min_score": float(followup_min_score),
        "followup_max_symbols": int(followup_max_symbols),
        "followup_candidate_count": int(len({str(row.get("symbol") or "") for row in followup_candidates if row.get("symbol")} - current_set)),
        "followup_selected_count": int(source_counts.get("followup", 0)),
        "collection_symbol_count": int(len(frame)),
        "max_total_symbols": int(max_total_symbols),
        "require_data_quality": bool(require_data_quality),
        "include_confidence": sorted(include_confidence),
        "stage_keywords": sorted(stage_keywords),
        "source_counts": {str(key): int(value) for key, value in source_counts.items()},
        "followup_origin_date_counts": {str(key): int(value) for key, value in origin_counts.items()},
        "symbols_preview": frame.head(20).to_dict("records") if not frame.empty else [],
    }
    return frame, status


def write_collection_outputs(
    base_dir: str | Path,
    *,
    date_value: str,
    collection: pd.DataFrame,
    status: dict[str, object],
    write_latest: bool = True,
) -> dict[str, Path]:
    base = Path(base_dir).expanduser()
    output_dir = base / "universe" / f"date={date_value}"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "us_microstructure_collection_universe.csv"
    txt_path = output_dir / "us_microstructure_collection_universe.txt"
    status_path = output_dir / "collection_status.json"
    collection.to_csv(csv_path, index=False)
    symbols = collection["symbol"].dropna().astype(str).tolist() if "symbol" in collection.columns else []
    txt_path.write_text("\n".join(symbols) + ("\n" if symbols else ""), encoding="utf-8")
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    outputs = {"collection_csv": csv_path, "collection_txt": txt_path, "collection_status": status_path}
    if write_latest:
        latest_dir = base / "universe"
        latest_dir.mkdir(parents=True, exist_ok=True)
        latest_csv = latest_dir / "us_microstructure_collection_universe_latest.csv"
        latest_txt = latest_dir / "us_microstructure_collection_universe_latest.txt"
        latest_status = latest_dir / "us_microstructure_collection_universe_status_latest.json"
        collection.to_csv(latest_csv, index=False)
        latest_txt.write_text("\n".join(symbols) + ("\n" if symbols else ""), encoding="utf-8")
        latest_status.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        outputs.update(
            {
                "collection_latest_csv": latest_csv,
                "collection_latest_txt": latest_txt,
                "collection_status_latest": latest_status,
            }
        )
    return outputs


def _sync_outputs(paths: Iterable[Path], *, base_dir: Path, nas_host: str, nas_dir: str) -> list[dict[str, str]]:
    return _sync_paths_to_nas(paths, local_base=base_dir, nas_host=nas_host, nas_dir=nas_dir)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build rolling US microstructure collection universe.")
    parser.add_argument("--base-dir", default=os.environ.get("US_MICROSTRUCTURE_DIR", str(DEFAULT_BASE_DIR)))
    parser.add_argument("--date", default=os.environ.get("US_MICROSTRUCTURE_COLLECTION_DATE", _collection_date_from_utc()))
    parser.add_argument(
        "--current-universe-file",
        default=os.environ.get(
            "US_MICROSTRUCTURE_CURRENT_UNIVERSE_FILE",
            str(Path(os.environ.get("US_MICROSTRUCTURE_DIR", str(DEFAULT_BASE_DIR))) / "universe" / "us_microstructure_candidates_latest.txt"),
        ),
    )
    parser.add_argument("--followup-days", type=int, default=int(os.environ.get("US_MICROSTRUCTURE_FOLLOWUP_DAYS", str(DEFAULT_FOLLOWUP_DAYS))))
    parser.add_argument("--followup-max-symbols", type=int, default=int(os.environ.get("US_MICROSTRUCTURE_FOLLOWUP_MAX_SYMBOLS", str(DEFAULT_FOLLOWUP_MAX_SYMBOLS))))
    parser.add_argument("--followup-min-score", type=float, default=float(os.environ.get("US_MICROSTRUCTURE_FOLLOWUP_MIN_SCORE", str(DEFAULT_FOLLOWUP_MIN_SCORE))))
    parser.add_argument("--max-total-symbols", type=int, default=int(os.environ.get("US_MICROSTRUCTURE_COLLECTION_MAX_SYMBOLS", str(DEFAULT_MAX_TOTAL_SYMBOLS))))
    parser.add_argument("--followup-confidence", default=os.environ.get("US_MICROSTRUCTURE_FOLLOWUP_CONFIDENCE", DEFAULT_FOLLOWUP_CONFIDENCE))
    parser.add_argument("--followup-stage-keywords", default=os.environ.get("US_MICROSTRUCTURE_FOLLOWUP_STAGE_KEYWORDS", DEFAULT_FOLLOWUP_STAGE_KEYWORDS))
    parser.add_argument("--allow-low-quality", action="store_true")
    parser.add_argument("--nas-host", default=os.environ.get("US_MICROSTRUCTURE_NAS_HOST", ""))
    parser.add_argument("--nas-dir", default=os.environ.get("US_MICROSTRUCTURE_NAS_DIR", DEFAULT_NAS_DIR))
    parser.add_argument("--no-nas-sync", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    base_dir = Path(args.base_dir).expanduser()
    collection, status = build_collection_universe(
        base_dir=base_dir,
        date_value=args.date[:10],
        current_universe_file=args.current_universe_file,
        followup_days=args.followup_days,
        followup_max_symbols=args.followup_max_symbols,
        followup_min_score=args.followup_min_score,
        max_total_symbols=args.max_total_symbols,
        followup_confidence=args.followup_confidence,
        followup_stage_keywords=args.followup_stage_keywords,
        require_data_quality=not bool(args.allow_low_quality),
    )
    outputs = write_collection_outputs(base_dir, date_value=args.date[:10], collection=collection, status=status)
    if not args.no_nas_sync:
        nas_results = _sync_outputs(outputs.values(), base_dir=base_dir, nas_host=args.nas_host, nas_dir=args.nas_dir)
        if nas_results:
            status["nas_sync"] = nas_results
            outputs = write_collection_outputs(base_dir, date_value=args.date[:10], collection=collection, status=status)
            _sync_outputs(
                [outputs["collection_status"], outputs.get("collection_status_latest", outputs["collection_status"])],
                base_dir=base_dir,
                nas_host=args.nas_host,
                nas_dir=args.nas_dir,
            )

    print(
        "Built US microstructure collection universe: total={total} current={current} followup={followup}".format(
            total=int(status.get("collection_symbol_count") or 0),
            current=int(status.get("current_symbol_count") or 0),
            followup=int(status.get("followup_selected_count") or 0),
        )
    )
    print(f"Wrote collection universe: {outputs['collection_latest_txt']}")
    return 0 if not collection.empty else 1


if __name__ == "__main__":
    raise SystemExit(main())
