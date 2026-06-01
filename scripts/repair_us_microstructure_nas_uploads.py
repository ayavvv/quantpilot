"""Repair failed US microstructure NAS uploads recorded in manifest files."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.collect_us_microstructure import DEFAULT_NAS_DIR, _copy_to_nas


DATA_DIR = Path(os.environ.get("DATA_DIR", str(Path.home() / "quantpilot_data")))
DEFAULT_BASE_DIR = Path(os.environ.get("US_MICROSTRUCTURE_DIR", str(DATA_DIR / "us_microstructure")))


def _date_default() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _manifest_paths(base_dir: Path, date: str) -> list[Path]:
    manifest_dir = base_dir / "manifests" / f"date={date}"
    if not manifest_dir.exists():
        return []
    return sorted(manifest_dir.glob("manifest-*.jsonl"))


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            payload = {
                "kind": "manifest_error",
                "nas_upload_status": "failed",
                "nas_error": f"invalid JSON: {exc}",
            }
        if isinstance(payload, dict):
            records.append(payload)
    return records


def _write_manifest(path: Path, records: list[dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    tmp.replace(path)


def _needs_repair(record: dict[str, Any]) -> bool:
    return str(record.get("nas_upload_status") or "").lower() != "ok"


def repair_manifest_uploads(
    *,
    base_dir: str | Path,
    date: str,
    nas_host: str,
    nas_dir: str,
    dry_run: bool = False,
    limit: int = 0,
) -> dict[str, Any]:
    base = Path(base_dir).expanduser()
    checked = 0
    repaired = 0
    missing_local = 0
    failed = 0
    manifest_sync_failed = 0
    changed_paths: list[str] = []
    records_by_status: dict[str, int] = {}

    for manifest_path in _manifest_paths(base, date):
        records = _read_manifest(manifest_path)
        changed = False
        for record in records:
            if not _needs_repair(record):
                continue
            if limit and checked >= int(limit):
                break
            checked += 1
            local_path = Path(str(record.get("local_path") or "")).expanduser()
            previous_status = str(record.get("nas_upload_status") or "")
            records_by_status[previous_status or "missing"] = records_by_status.get(previous_status or "missing", 0) + 1
            if not local_path.exists():
                missing_local += 1
                record["nas_upload_status"] = "failed"
                record["nas_error"] = f"local file missing during repair: {local_path}"
                record["repair_checked_at"] = _utc_now_iso()
                changed = True
                continue
            if dry_run:
                continue
            if not nas_host or not nas_dir:
                failed += 1
                record["nas_upload_status"] = "failed"
                record["nas_error"] = "repair requires --nas-host and --nas-dir"
                record["repair_checked_at"] = _utc_now_iso()
                changed = True
                continue
            status, remote_path, error = _copy_to_nas(local_path, base, nas_host, nas_dir)
            if status == "ok":
                repaired += 1
                record["previous_nas_upload_status"] = previous_status
                record["previous_nas_error"] = str(record.get("nas_error") or "")
                record["nas_upload_status"] = status
                record["nas_path"] = remote_path
                record["nas_error"] = ""
                record["repaired_at"] = _utc_now_iso()
            else:
                failed += 1
                record["nas_upload_status"] = status
                record["nas_path"] = remote_path
                record["nas_error"] = error
                record["repair_checked_at"] = _utc_now_iso()
            changed = True
        if changed and not dry_run:
            _write_manifest(manifest_path, records)
            changed_paths.append(str(manifest_path))
            status, _, _ = _copy_to_nas(manifest_path, base, nas_host, nas_dir)
            if status != "ok":
                manifest_sync_failed += 1
        if limit and checked >= int(limit):
            break

    return {
        "date": date,
        "base_dir": str(base),
        "dry_run": dry_run,
        "checked": checked,
        "repaired": repaired,
        "missing_local": missing_local,
        "failed": failed,
        "manifest_sync_failed": manifest_sync_failed,
        "records_by_previous_status": records_by_status,
        "changed_manifest_paths": changed_paths,
        "ok": failed == 0 and missing_local == 0 and manifest_sync_failed == 0,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repair failed US microstructure NAS uploads from manifest rows.")
    parser.add_argument("--base-dir", default=os.environ.get("US_MICROSTRUCTURE_DIR", str(DEFAULT_BASE_DIR)))
    parser.add_argument("--date", default=os.environ.get("US_MICROSTRUCTURE_DATE", _date_default()))
    parser.add_argument("--nas-host", default=os.environ.get("US_MICROSTRUCTURE_NAS_HOST", ""))
    parser.add_argument("--nas-dir", default=os.environ.get("US_MICROSTRUCTURE_NAS_DIR", DEFAULT_NAS_DIR))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = repair_manifest_uploads(
        base_dir=args.base_dir,
        date=args.date,
        nas_host=str(args.nas_host or ""),
        nas_dir=str(args.nas_dir or ""),
        dry_run=bool(args.dry_run),
        limit=int(args.limit or 0),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
