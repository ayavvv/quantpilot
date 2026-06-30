"""Prune local runtime logs that are not part of trading data state."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_FUTU_LOG_DIR = Path(os.environ.get("FUTU_OPEND_LOG_DIR", str(Path.home() / ".com.futunn.FutuOpenD" / "Log")))
DEFAULT_QUANTPILOT_LOG_DIR = PROJECT_DIR / "logs"
FUTU_LOG_SUFFIXES = (".log", ".logs", ".ftlog")


def _file_size(path: Path) -> int:
    try:
        return int(path.stat().st_size)
    except OSError:
        return 0


def _older_than(path: Path, cutoff: datetime) -> bool:
    try:
        modified = datetime.fromtimestamp(path.stat().st_mtime)
    except OSError:
        return False
    return modified < cutoff


def _is_futu_log_file(path: Path) -> bool:
    name = path.name
    return name.endswith(FUTU_LOG_SUFFIXES) or any(f"{suffix}." in name for suffix in FUTU_LOG_SUFFIXES)


def _discover_futu_logs(log_dir: Path, cutoff: datetime) -> list[Path]:
    if not log_dir.exists():
        return []
    return sorted(
        path
        for path in log_dir.iterdir()
        if path.is_file() and _is_futu_log_file(path) and _older_than(path, cutoff)
    )


def _discover_quantpilot_logs(log_dir: Path, cutoff: datetime) -> list[Path]:
    if not log_dir.exists():
        return []
    return sorted(path for path in log_dir.rglob("*.log") if path.is_file() and _older_than(path, cutoff))


def _delete_files(paths: Iterable[Path]) -> tuple[int, int, list[str]]:
    count = 0
    bytes_deleted = 0
    errors: list[str] = []
    for path in paths:
        size = _file_size(path)
        try:
            path.unlink()
            count += 1
            bytes_deleted += size
        except FileNotFoundError:
            pass
        except OSError as exc:
            errors.append(f"{path}: {exc}")
    return count, bytes_deleted, errors


def cleanup_local_runtime_logs(
    *,
    futu_log_dir: str | Path,
    futu_retention_days: int,
    quantpilot_log_dir: str | Path,
    quantpilot_retention_days: int,
    now: str = "",
    execute: bool = False,
) -> dict[str, object]:
    timestamp = datetime.fromisoformat(now) if now else datetime.now()
    futu_cutoff = timestamp - timedelta(days=max(0, int(futu_retention_days)))
    quantpilot_cutoff = timestamp - timedelta(days=max(0, int(quantpilot_retention_days)))
    futu_dir = Path(futu_log_dir).expanduser()
    quantpilot_dir = Path(quantpilot_log_dir).expanduser()
    futu_candidates = _discover_futu_logs(futu_dir, futu_cutoff)
    quantpilot_candidates = _discover_quantpilot_logs(quantpilot_dir, quantpilot_cutoff)
    all_candidates = futu_candidates + quantpilot_candidates
    planned_bytes = sum(_file_size(path) for path in all_candidates)

    deleted_count = 0
    deleted_bytes = 0
    errors: list[str] = []
    if execute:
        deleted_count, deleted_bytes, errors = _delete_files(all_candidates)

    return {
        "execute": bool(execute),
        "now": timestamp.isoformat(timespec="seconds"),
        "futu_log_dir": str(futu_dir),
        "futu_retention_days": int(futu_retention_days),
        "futu_candidate_count": len(futu_candidates),
        "quantpilot_log_dir": str(quantpilot_dir),
        "quantpilot_retention_days": int(quantpilot_retention_days),
        "quantpilot_candidate_count": len(quantpilot_candidates),
        "planned_file_count": len(all_candidates),
        "planned_bytes": planned_bytes,
        "deleted_file_count": deleted_count,
        "deleted_bytes": deleted_bytes,
        "errors": errors,
        "ok": not errors,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean local QuantPilot/Futu runtime logs.")
    parser.add_argument("--futu-log-dir", default=os.environ.get("FUTU_OPEND_LOG_DIR", str(DEFAULT_FUTU_LOG_DIR)))
    parser.add_argument(
        "--futu-retention-days",
        type=int,
        default=int(os.environ.get("FUTU_OPEND_LOG_RETENTION_DAYS", "14")),
    )
    parser.add_argument("--quantpilot-log-dir", default=os.environ.get("QUANTPILOT_LOG_DIR", str(DEFAULT_QUANTPILOT_LOG_DIR)))
    parser.add_argument(
        "--quantpilot-retention-days",
        type=int,
        default=int(os.environ.get("QUANTPILOT_LOG_RETENTION_DAYS", "30")),
    )
    parser.add_argument("--now", default=os.environ.get("LOCAL_RUNTIME_LOG_CLEANUP_NOW", ""))
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = cleanup_local_runtime_logs(
        futu_log_dir=args.futu_log_dir,
        futu_retention_days=int(args.futu_retention_days),
        quantpilot_log_dir=args.quantpilot_log_dir,
        quantpilot_retention_days=int(args.quantpilot_retention_days),
        now=str(args.now or ""),
        execute=bool(args.execute),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
