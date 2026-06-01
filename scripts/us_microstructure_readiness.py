"""Readiness checks for US microstructure collection, validation, and reporting."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from scripts.collect_us_microstructure import DEFAULT_NAS_DIR, _copy_to_nas


DATA_DIR = Path(os.environ.get("DATA_DIR", str(Path.home() / "quantpilot_data")))
DEFAULT_BASE_DIR = Path(os.environ.get("US_MICROSTRUCTURE_DIR", str(DATA_DIR / "us_microstructure")))
LAUNCHD_LABELS = (
    "com.quantpilot.us_microstructure.collect",
    "com.quantpilot.us_microstructure.report",
)


def _read_json(path: Path) -> tuple[dict[str, Any], str]:
    if not path.exists():
        return {}, f"missing: {path}"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {}, str(exc)
    if not isinstance(payload, dict):
        return {}, "not a JSON object"
    return payload, ""


def _date_default() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def load_manifest_records(base_dir: str | Path, *, date: str, latest_only: bool = True) -> list[dict[str, Any]]:
    manifest_dir = Path(base_dir).expanduser() / "manifests" / f"date={date}"
    records: list[dict[str, Any]] = []
    if not manifest_dir.exists():
        return records
    manifest_paths = sorted(manifest_dir.glob("manifest-*.jsonl"))
    if latest_only and manifest_paths:
        manifest_paths = [manifest_paths[-1]]
    for path in manifest_paths:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                payload = {"kind": "manifest_error", "nas_upload_status": "failed", "nas_error": "invalid JSON"}
            if isinstance(payload, dict):
                records.append(payload)
    return records


def check_manifest(base_dir: str | Path, *, date: str, min_records: int = 1, latest_only: bool = True) -> dict[str, Any]:
    records = load_manifest_records(base_dir, date=date, latest_only=latest_only)
    status_counts: dict[str, int] = {}
    kind_counts: dict[str, int] = {}
    row_counts: dict[str, int] = {}
    for record in records:
        status = str(record.get("nas_upload_status") or "")
        kind = str(record.get("kind") or "")
        status_counts[status] = status_counts.get(status, 0) + 1
        kind_counts[kind] = kind_counts.get(kind, 0) + 1
        row_counts[kind] = row_counts.get(kind, 0) + int(record.get("row_count") or 0)
    issues = []
    if len(records) < int(min_records):
        issues.append(f"manifest records below minimum: {len(records)} < {min_records}")
    failed = status_counts.get("failed", 0)
    if failed:
        issues.append(f"manifest contains failed NAS uploads: {failed}")
    return {
        "ok": not issues,
        "date": date,
        "latest_only": latest_only,
        "manifest_count": len(records),
        "status_counts": status_counts,
        "kind_counts": kind_counts,
        "row_counts": row_counts,
        "issues": issues,
    }


def check_prices(base_dir: str | Path) -> dict[str, Any]:
    path = Path(base_dir).expanduser() / "validation" / "prices" / "us_daily_prices_status.json"
    payload, error = _read_json(path)
    issues = []
    if error:
        issues.append(f"price status unreadable: {error}")
    elif str(payload.get("status") or "").lower() not in {"ok", "partial"}:
        issues.append(f"price status not ok/partial: {payload.get('status')}")
    if payload and int(payload.get("price_row_count") or 0) <= 0:
        issues.append("price status has zero rows")
    return {
        "ok": not issues,
        "path": str(path),
        "exists": path.exists(),
        "status": payload.get("status", "") if payload else "",
        "symbol_count": int(payload.get("symbol_count") or 0) if payload else 0,
        "price_row_count": int(payload.get("price_row_count") or 0) if payload else 0,
        "errors": payload.get("errors", {}) if payload else {},
        "issues": issues,
    }


def check_validation_gate(base_dir: str | Path) -> dict[str, Any]:
    path = Path(base_dir).expanduser() / "validation" / "active_gate.json"
    payload, error = _read_json(path)
    issues = []
    if error:
        issues.append(f"validation gate unreadable: {error}")
    return {
        "ok": not issues,
        "path": str(path),
        "exists": path.exists(),
        "state": str(payload.get("state") or "") if payload else "",
        "validated": bool(payload.get("validated")) if payload else False,
        "validated_sides": payload.get("validated_sides", {}) if payload else {},
        "event_count": int(payload.get("event_count") or 0) if payload else 0,
        "forward_return_count": int(payload.get("forward_return_count") or 0) if payload else 0,
        "reason": str(payload.get("reason") or "") if payload else "",
        "issues": issues,
    }


def check_report(base_dir: str | Path, *, date: str) -> dict[str, Any]:
    base = Path(base_dir).expanduser()
    status_path = base / "reports" / f"date={date}" / "status.json"
    html_path = base / "reports" / f"date={date}" / "us_microstructure_flow_report.html"
    latest_html = base / "reports" / "us_microstructure_flow_report_latest.html"
    payload, error = _read_json(status_path)
    issues = []
    if error:
        issues.append(f"report status unreadable: {error}")
    if not html_path.exists():
        issues.append(f"report HTML missing: {html_path}")
    if not latest_html.exists():
        issues.append(f"latest report HTML missing: {latest_html}")
    high_count = int(payload.get("high_count") or 0) if payload else 0
    data_quality = payload.get("data_quality", {}) if payload else {}
    data_quality_ready = bool(isinstance(data_quality, dict) and data_quality.get("high_confidence_data_quality_ok"))
    if high_count > 0 and not (isinstance(data_quality, dict) and data_quality.get("high_confidence_data_quality_ok")):
        issues.append("report has high-confidence signals without a passing data-quality gate")
    return {
        "ok": not issues,
        "status_path": str(status_path),
        "html_path": str(html_path),
        "latest_html_path": str(latest_html),
        "exists": status_path.exists() and html_path.exists(),
        "signal_count": int(payload.get("signal_count") or 0) if payload else 0,
        "high_count": high_count,
        "watch_count": int(payload.get("watch_count") or 0) if payload else 0,
        "data_quality": data_quality if isinstance(data_quality, dict) else {},
        "data_quality_ready": data_quality_ready,
        "issues": issues,
    }


def _launchctl_print(label: str) -> tuple[int, str]:
    uid = str(os.getuid())
    result = subprocess.run(
        ["launchctl", "print", f"gui/{uid}/{label}"],
        text=True,
        capture_output=True,
        check=False,
    )
    return result.returncode, result.stdout + result.stderr


def check_launchd(
    *,
    runner: Callable[[str], tuple[int, str]] | None = None,
    labels: tuple[str, ...] = LAUNCHD_LABELS,
) -> dict[str, Any]:
    run = runner or _launchctl_print
    services: dict[str, dict[str, Any]] = {}
    issues = []
    for label in labels:
        code, output = run(label)
        loaded = code == 0 and "state =" in output
        services[label] = {
            "loaded": loaded,
            "state": _extract_launchd_state(output),
            "runs": _extract_launchd_runs(output),
        }
        if not loaded:
            issues.append(f"launchd service not loaded: {label}")
    return {"ok": not issues, "services": services, "issues": issues}


def _extract_launchd_state(output: str) -> str:
    for line in output.splitlines():
        line = line.strip()
        if line.startswith("state ="):
            return line.split("=", 1)[1].strip()
    return ""


def _extract_launchd_runs(output: str) -> int:
    for line in output.splitlines():
        line = line.strip()
        if line.startswith("runs ="):
            try:
                return int(line.split("=", 1)[1].strip())
            except ValueError:
                return 0
    return 0


def build_readiness_snapshot(
    *,
    base_dir: str | Path,
    date: str,
    min_manifest_records: int = 1,
    latest_manifest_only: bool = True,
    include_launchd: bool = True,
    launchd_runner: Callable[[str], tuple[int, str]] | None = None,
) -> dict[str, Any]:
    checks: dict[str, Any] = {
        "manifest": check_manifest(base_dir, date=date, min_records=min_manifest_records, latest_only=latest_manifest_only),
        "prices": check_prices(base_dir),
        "validation_gate": check_validation_gate(base_dir),
        "report": check_report(base_dir, date=date),
    }
    if include_launchd:
        checks["launchd"] = check_launchd(runner=launchd_runner)
    validation_ready = bool(checks["validation_gate"].get("validated"))
    data_quality_ready = bool(checks["report"].get("data_quality_ready"))
    high_confidence_ready = validation_ready and data_quality_ready
    issues = []
    for name, payload in checks.items():
        for issue in payload.get("issues", []):
            issues.append(f"{name}: {issue}")
    return {
        "ok": not issues,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "base_dir": str(Path(base_dir).expanduser()),
        "date": date,
        "high_confidence_ready": high_confidence_ready,
        "high_confidence_requirements": {
            "validation_gate_validated": validation_ready,
            "data_quality_gate_ready": data_quality_ready,
        },
        "checks": checks,
        "issues": issues,
    }


def write_readiness_snapshot(base_dir: str | Path, snapshot: dict[str, Any]) -> Path:
    output_dir = Path(base_dir).expanduser() / "readiness"
    output_dir.mkdir(parents=True, exist_ok=True)
    date = str(snapshot.get("date") or _date_default())
    dated_path = output_dir / f"us_microstructure_readiness_{date.replace('-', '')}.json"
    latest_path = output_dir / "us_microstructure_readiness_latest.json"
    text = json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n"
    dated_path.write_text(text, encoding="utf-8")
    latest_path.write_text(text, encoding="utf-8")
    return dated_path


def _readiness_output_paths(base_dir: str | Path, date: str) -> list[Path]:
    output_dir = Path(base_dir).expanduser() / "readiness"
    return [
        output_dir / f"us_microstructure_readiness_{date.replace('-', '')}.json",
        output_dir / "us_microstructure_readiness_latest.json",
    ]


def sync_readiness_outputs(paths: list[Path], *, base_dir: str | Path, nas_host: str, nas_dir: str) -> list[dict[str, str]]:
    results: list[dict[str, str]] = []
    if not nas_host or not nas_dir:
        return results
    local_base = Path(base_dir).expanduser()
    for path in paths:
        status, remote_path, error = _copy_to_nas(path, local_base, nas_host, nas_dir)
        results.append({"local_path": str(path), "nas_path": remote_path, "status": status, "error": error})
    return results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check US microstructure pipeline readiness.")
    parser.add_argument("--base-dir", default=os.environ.get("US_MICROSTRUCTURE_DIR", str(DEFAULT_BASE_DIR)))
    parser.add_argument("--date", default=os.environ.get("US_MICROSTRUCTURE_DATE", _date_default()))
    parser.add_argument("--min-manifest-records", type=int, default=int(os.environ.get("US_MICROSTRUCTURE_MIN_MANIFEST_RECORDS", "1")))
    parser.add_argument("--nas-host", default=os.environ.get("US_MICROSTRUCTURE_NAS_HOST", ""))
    parser.add_argument("--nas-dir", default=os.environ.get("US_MICROSTRUCTURE_NAS_DIR", DEFAULT_NAS_DIR))
    parser.add_argument("--no-nas-sync", action="store_true")
    parser.add_argument("--all-manifests", action="store_true")
    parser.add_argument("--skip-launchd", action="store_true")
    parser.add_argument("--write-json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    snapshot = build_readiness_snapshot(
        base_dir=args.base_dir,
        date=args.date,
        min_manifest_records=args.min_manifest_records,
        latest_manifest_only=not args.all_manifests,
        include_launchd=not args.skip_launchd,
    )
    nas_results = []
    if args.write_json:
        path = write_readiness_snapshot(args.base_dir, snapshot)
        print(f"Wrote readiness: {path}")
        if not args.no_nas_sync:
            output_paths = _readiness_output_paths(args.base_dir, str(snapshot.get("date") or args.date))
            nas_results = sync_readiness_outputs(
                output_paths,
                base_dir=args.base_dir,
                nas_host=str(args.nas_host or ""),
                nas_dir=str(args.nas_dir or ""),
            )
            if nas_results:
                print(json.dumps({"readiness_nas_sync": nas_results}, ensure_ascii=False, indent=2))
    print(json.dumps(snapshot, ensure_ascii=False, indent=2))
    nas_ok = all(item.get("status") == "ok" for item in nas_results)
    return 0 if snapshot["ok"] and nas_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
