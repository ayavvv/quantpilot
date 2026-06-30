"""Prune local US microstructure raw parquet after NAS archival.

The cleanup is intentionally conservative:

* only raw ``trades``/``order_book``/``quotes`` date partitions are eligible;
* generated features, signals, quality, validation, reports, and manifests stay local;
* a date is deleted only when every local raw file is covered by an ``ok`` NAS
  upload manifest row, unless partial cleanup is explicitly allowed.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import date as date_type
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable


DATA_DIR = Path(os.environ.get("DATA_DIR", str(Path.home() / "quantpilot_data")))
DEFAULT_BASE_DIR = Path(os.environ.get("US_MICROSTRUCTURE_DIR", str(DATA_DIR / "us_microstructure")))
RAW_KINDS = ("trades", "order_book", "quotes")


def _today(value: str = "") -> date_type:
    if value:
        return datetime.strptime(value, "%Y-%m-%d").date()
    return datetime.now().date()


def _date_from_partition(path: Path) -> str:
    if not path.name.startswith("date="):
        return ""
    value = path.name.split("=", 1)[1][:10]
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        return ""
    return value


def discover_raw_dates(base_dir: str | Path, *, kinds: Iterable[str] = RAW_KINDS) -> list[str]:
    base = Path(base_dir).expanduser()
    dates: set[str] = set()
    for kind in kinds:
        root = base / kind
        if not root.exists():
            continue
        for path in root.glob("date=*"):
            if path.is_dir():
                value = _date_from_partition(path)
                if value:
                    dates.add(value)
    return sorted(dates)


def _manifest_paths(base: Path, day: str) -> list[Path]:
    root = base / "manifests" / f"date={day}"
    if not root.exists():
        return []
    return sorted(root.glob("manifest-*.jsonl"))


def _read_manifest_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            payload = {"kind": "manifest_error", "nas_upload_status": "failed", "nas_error": str(exc)}
        if isinstance(payload, dict):
            records.append(payload)
    return records


def _is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
    except ValueError:
        return False
    return True


def _expected_raw_root(base: Path, kind: str, day: str) -> Path:
    return base / kind / f"date={day}"


def _is_expected_raw_path(path: Path, *, base: Path, kind: str, day: str) -> bool:
    root = _expected_raw_root(base, kind, day)
    return _is_relative_to(path, root) and path.suffix == ".parquet"


def _local_raw_files(base: Path, day: str, *, kinds: Iterable[str]) -> list[Path]:
    files: list[Path] = []
    for kind in kinds:
        root = _expected_raw_root(base, kind, day)
        if root.exists():
            files.extend(sorted(path for path in root.rglob("*.parquet") if path.is_file()))
    return files


def _file_size(path: Path) -> int:
    try:
        return int(path.stat().st_size)
    except OSError:
        return 0


def _cleanup_empty_dirs(base: Path, day: str, *, kinds: Iterable[str]) -> int:
    removed = 0
    for kind in kinds:
        root = _expected_raw_root(base, kind, day)
        if not root.exists():
            continue
        for path in sorted((p for p in root.rglob("*") if p.is_dir()), key=lambda item: len(item.parts), reverse=True):
            try:
                path.rmdir()
                removed += 1
            except OSError:
                pass
        try:
            root.rmdir()
            removed += 1
        except OSError:
            pass
    return removed


def _mount_target_for(local_path: Path, *, local_base: Path, nas_mount_dir: Path) -> Path:
    return nas_mount_dir / local_path.relative_to(local_base)


def _manifest_coverage(
    base: Path,
    day: str,
    *,
    kinds: Iterable[str],
) -> dict[str, Any]:
    kind_set = set(kinds)
    ok_paths: dict[Path, dict[str, Any]] = {}
    bad_records: list[dict[str, Any]] = []
    ignored_records = 0
    total_records = 0

    for manifest_path in _manifest_paths(base, day):
        for record in _read_manifest_records(manifest_path):
            total_records += 1
            kind = str(record.get("kind") or "")
            if kind not in kind_set:
                ignored_records += 1
                continue
            local_path = Path(str(record.get("local_path") or "")).expanduser()
            if not _is_expected_raw_path(local_path, base=base, kind=kind, day=day):
                item = dict(record)
                item["cleanup_skip_reason"] = "manifest local_path is outside expected raw date partition"
                bad_records.append(item)
                continue
            status = str(record.get("nas_upload_status") or "").lower()
            if status == "ok":
                ok_paths[local_path] = record
            else:
                bad_records.append(record)

    return {
        "total_manifest_records": total_records,
        "ignored_manifest_records": ignored_records,
        "ok_paths": ok_paths,
        "bad_records": bad_records,
    }


def plan_date_cleanup(
    base_dir: str | Path,
    day: str,
    *,
    kinds: Iterable[str] = RAW_KINDS,
    allow_partial: bool = False,
    nas_mount_dir: str | Path = "",
    verify_nas_mount: bool = False,
) -> dict[str, Any]:
    base = Path(base_dir).expanduser()
    local_files = _local_raw_files(base, day, kinds=kinds)
    coverage = _manifest_coverage(base, day, kinds=kinds)
    ok_paths: dict[Path, dict[str, Any]] = coverage["ok_paths"]
    bad_records: list[dict[str, Any]] = coverage["bad_records"]
    unarchived_files = [path for path in local_files if path not in ok_paths]
    candidate_files = [path for path in local_files if path in ok_paths]
    missing_mount_files: list[str] = []

    if verify_nas_mount:
        mount = Path(nas_mount_dir).expanduser() if nas_mount_dir else Path()
        if not mount.is_dir():
            missing_mount_files = [f"NAS mount directory is not available: {mount}"]
        else:
            for path in candidate_files:
                if not _mount_target_for(path, local_base=base, nas_mount_dir=mount).exists():
                    missing_mount_files.append(str(path))

    issues: list[str] = []
    if not local_files:
        issues.append("no local raw files for date")
    if coverage["total_manifest_records"] == 0:
        issues.append("no manifest records for date")
    if bad_records and not allow_partial:
        issues.append(f"manifest has non-ok or invalid raw records: {len(bad_records)}")
    if unarchived_files and not allow_partial:
        issues.append(f"local raw files without ok NAS manifest rows: {len(unarchived_files)}")
    if missing_mount_files:
        issues.append(f"NAS mount verification failed: {len(missing_mount_files)} missing")

    bytes_to_delete = sum(_file_size(path) for path in candidate_files)
    return {
        "date": day,
        "eligible": not issues and bool(candidate_files),
        "candidate_file_count": len(candidate_files),
        "candidate_bytes": bytes_to_delete,
        "local_raw_file_count": len(local_files),
        "bad_manifest_record_count": len(bad_records),
        "unarchived_file_count": len(unarchived_files),
        "missing_mount_file_count": len(missing_mount_files),
        "issues": issues,
        "files": [str(path) for path in candidate_files],
    }


def _eligible_dates(base: Path, *, retention_days: int, today: date_type, dates: Iterable[str]) -> list[str]:
    cutoff = today - timedelta(days=max(0, int(retention_days)))
    result: list[str] = []
    for day in dates:
        parsed = datetime.strptime(day, "%Y-%m-%d").date()
        if parsed <= cutoff:
            result.append(day)
    return result


def cleanup_us_microstructure_archive(
    *,
    base_dir: str | Path,
    retention_days: int,
    today: str = "",
    dates: Iterable[str] | None = None,
    kinds: Iterable[str] = RAW_KINDS,
    execute: bool = False,
    allow_partial: bool = False,
    max_delete_bytes: int = 0,
    nas_mount_dir: str | Path = "",
    verify_nas_mount: bool = False,
) -> dict[str, Any]:
    base = Path(base_dir).expanduser()
    today_value = _today(today)
    discovered_dates = sorted(dates) if dates is not None else discover_raw_dates(base, kinds=kinds)
    candidates = _eligible_dates(base, retention_days=retention_days, today=today_value, dates=discovered_dates)
    date_plans = [
        plan_date_cleanup(
            base,
            day,
            kinds=kinds,
            allow_partial=allow_partial,
            nas_mount_dir=nas_mount_dir,
            verify_nas_mount=verify_nas_mount,
        )
        for day in candidates
    ]
    eligible_plans = [item for item in date_plans if item["eligible"]]
    total_bytes = sum(int(item["candidate_bytes"]) for item in eligible_plans)
    total_files = sum(int(item["candidate_file_count"]) for item in eligible_plans)
    blocked_by_cap = bool(max_delete_bytes and total_bytes > int(max_delete_bytes))

    deleted_files = 0
    deleted_bytes = 0
    removed_empty_dirs = 0
    errors: list[str] = []
    if execute and not blocked_by_cap:
        for item in eligible_plans:
            for raw_path in item["files"]:
                path = Path(raw_path)
                size = _file_size(path)
                try:
                    path.unlink()
                    deleted_files += 1
                    deleted_bytes += size
                except FileNotFoundError:
                    pass
                except OSError as exc:
                    errors.append(f"{path}: {exc}")
            removed_empty_dirs += _cleanup_empty_dirs(base, str(item["date"]), kinds=kinds)

    return {
        "base_dir": str(base),
        "today": today_value.isoformat(),
        "retention_days": int(retention_days),
        "execute": bool(execute),
        "allow_partial": bool(allow_partial),
        "verify_nas_mount": bool(verify_nas_mount),
        "nas_mount_dir": str(nas_mount_dir or ""),
        "discovered_dates": discovered_dates,
        "candidate_dates": candidates,
        "eligible_dates": [str(item["date"]) for item in eligible_plans],
        "skipped_dates": [
            {"date": str(item["date"]), "issues": item["issues"]}
            for item in date_plans
            if not item["eligible"]
        ],
        "planned_file_count": total_files,
        "planned_bytes": total_bytes,
        "max_delete_bytes": int(max_delete_bytes or 0),
        "blocked_by_cap": blocked_by_cap,
        "deleted_file_count": deleted_files,
        "deleted_bytes": deleted_bytes,
        "removed_empty_dir_count": removed_empty_dirs,
        "errors": errors,
        "ok": not errors and not blocked_by_cap,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean local US microstructure raw parquet after NAS archival.")
    parser.add_argument("--base-dir", default=os.environ.get("US_MICROSTRUCTURE_DIR", str(DEFAULT_BASE_DIR)))
    parser.add_argument(
        "--retention-days",
        type=int,
        default=int(os.environ.get("US_MICROSTRUCTURE_RAW_RETENTION_DAYS", "7")),
        help="Keep local raw parquet newer than this many calendar days.",
    )
    parser.add_argument("--today", default=os.environ.get("US_MICROSTRUCTURE_CLEANUP_TODAY", ""))
    parser.add_argument("--date", action="append", dest="dates", help="Limit cleanup planning to one date; repeatable.")
    parser.add_argument("--execute", action="store_true", help="Actually delete eligible local raw files.")
    parser.add_argument("--allow-partial", action="store_true", help="Delete ok-manifest files even if the date is partial.")
    parser.add_argument(
        "--max-delete-bytes",
        type=int,
        default=int(os.environ.get("US_MICROSTRUCTURE_CLEANUP_MAX_DELETE_BYTES", "0")),
    )
    parser.add_argument("--nas-mount-dir", default=os.environ.get("US_MICROSTRUCTURE_NAS_MOUNT_DIR", ""))
    parser.add_argument("--verify-nas-mount", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = cleanup_us_microstructure_archive(
        base_dir=args.base_dir,
        retention_days=int(args.retention_days),
        today=str(args.today or ""),
        dates=args.dates,
        execute=bool(args.execute),
        allow_partial=bool(args.allow_partial),
        max_delete_bytes=int(args.max_delete_bytes or 0),
        nas_mount_dir=str(args.nas_mount_dir or ""),
        verify_nas_mount=bool(args.verify_nas_mount),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
