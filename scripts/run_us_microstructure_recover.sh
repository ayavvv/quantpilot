#!/bin/bash
# Recovery guard for US microstructure jobs after reboot or missed schedules.

set -euo pipefail

export PATH="/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

EXTERNAL_DATA_DIR="${DATA_DIR-}"
EXTERNAL_US_MICROSTRUCTURE_DIR="${US_MICROSTRUCTURE_DIR-}"
EXTERNAL_US_MICROSTRUCTURE_REMOTE_HOST="${US_MICROSTRUCTURE_REMOTE_HOST-}"
EXTERNAL_US_MICROSTRUCTURE_REMOTE_HOSTS="${US_MICROSTRUCTURE_REMOTE_HOSTS-}"
EXTERNAL_US_MICROSTRUCTURE_RECOVER_NOW="${US_MICROSTRUCTURE_RECOVER_NOW-}"
EXTERNAL_US_MICROSTRUCTURE_RECOVER_DRY_RUN="${US_MICROSTRUCTURE_RECOVER_DRY_RUN-}"
EXTERNAL_US_MICROSTRUCTURE_RECOVER_COLLECT_START="${US_MICROSTRUCTURE_RECOVER_COLLECT_START-}"
EXTERNAL_US_MICROSTRUCTURE_RECOVER_COLLECT_END="${US_MICROSTRUCTURE_RECOVER_COLLECT_END-}"
EXTERNAL_US_MICROSTRUCTURE_RECOVER_REPORT_START="${US_MICROSTRUCTURE_RECOVER_REPORT_START-}"
EXTERNAL_US_MICROSTRUCTURE_RECOVER_REPORT_END="${US_MICROSTRUCTURE_RECOVER_REPORT_END-}"
EXTERNAL_US_MICROSTRUCTURE_RECOVER_MIN_COLLECT_SECONDS="${US_MICROSTRUCTURE_RECOVER_MIN_COLLECT_SECONDS-}"
EXTERNAL_US_MICROSTRUCTURE_RECOVER_REQUIRE_EMAIL="${US_MICROSTRUCTURE_RECOVER_REQUIRE_EMAIL-}"
EXTERNAL_US_MICROSTRUCTURE_RECOVER_SEND_EMAIL="${US_MICROSTRUCTURE_RECOVER_SEND_EMAIL-}"

if [ -f "$PROJECT_DIR/.env" ]; then
    set -a
    source "$PROJECT_DIR/.env"
    set +a
fi

[ -n "$EXTERNAL_DATA_DIR" ] && DATA_DIR="$EXTERNAL_DATA_DIR"
[ -n "$EXTERNAL_US_MICROSTRUCTURE_DIR" ] && US_MICROSTRUCTURE_DIR="$EXTERNAL_US_MICROSTRUCTURE_DIR"
[ -n "$EXTERNAL_US_MICROSTRUCTURE_REMOTE_HOST" ] && US_MICROSTRUCTURE_REMOTE_HOST="$EXTERNAL_US_MICROSTRUCTURE_REMOTE_HOST"
[ -n "$EXTERNAL_US_MICROSTRUCTURE_REMOTE_HOSTS" ] && US_MICROSTRUCTURE_REMOTE_HOSTS="$EXTERNAL_US_MICROSTRUCTURE_REMOTE_HOSTS"
[ -n "$EXTERNAL_US_MICROSTRUCTURE_RECOVER_NOW" ] && US_MICROSTRUCTURE_RECOVER_NOW="$EXTERNAL_US_MICROSTRUCTURE_RECOVER_NOW"
[ -n "$EXTERNAL_US_MICROSTRUCTURE_RECOVER_DRY_RUN" ] && US_MICROSTRUCTURE_RECOVER_DRY_RUN="$EXTERNAL_US_MICROSTRUCTURE_RECOVER_DRY_RUN"
[ -n "$EXTERNAL_US_MICROSTRUCTURE_RECOVER_COLLECT_START" ] && US_MICROSTRUCTURE_RECOVER_COLLECT_START="$EXTERNAL_US_MICROSTRUCTURE_RECOVER_COLLECT_START"
[ -n "$EXTERNAL_US_MICROSTRUCTURE_RECOVER_COLLECT_END" ] && US_MICROSTRUCTURE_RECOVER_COLLECT_END="$EXTERNAL_US_MICROSTRUCTURE_RECOVER_COLLECT_END"
[ -n "$EXTERNAL_US_MICROSTRUCTURE_RECOVER_REPORT_START" ] && US_MICROSTRUCTURE_RECOVER_REPORT_START="$EXTERNAL_US_MICROSTRUCTURE_RECOVER_REPORT_START"
[ -n "$EXTERNAL_US_MICROSTRUCTURE_RECOVER_REPORT_END" ] && US_MICROSTRUCTURE_RECOVER_REPORT_END="$EXTERNAL_US_MICROSTRUCTURE_RECOVER_REPORT_END"
[ -n "$EXTERNAL_US_MICROSTRUCTURE_RECOVER_MIN_COLLECT_SECONDS" ] && US_MICROSTRUCTURE_RECOVER_MIN_COLLECT_SECONDS="$EXTERNAL_US_MICROSTRUCTURE_RECOVER_MIN_COLLECT_SECONDS"
[ -n "$EXTERNAL_US_MICROSTRUCTURE_RECOVER_REQUIRE_EMAIL" ] && US_MICROSTRUCTURE_RECOVER_REQUIRE_EMAIL="$EXTERNAL_US_MICROSTRUCTURE_RECOVER_REQUIRE_EMAIL"
[ -n "$EXTERNAL_US_MICROSTRUCTURE_RECOVER_SEND_EMAIL" ] && US_MICROSTRUCTURE_RECOVER_SEND_EMAIL="$EXTERNAL_US_MICROSTRUCTURE_RECOVER_SEND_EMAIL"

DATA_DIR="${DATA_DIR:-$HOME/quantpilot_data}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}"
US_MICROSTRUCTURE_DIR="${US_MICROSTRUCTURE_DIR:-$DATA_DIR/us_microstructure}"
US_MICROSTRUCTURE_RECOVER_NOW="${US_MICROSTRUCTURE_RECOVER_NOW:-}"
US_MICROSTRUCTURE_RECOVER_DRY_RUN="${US_MICROSTRUCTURE_RECOVER_DRY_RUN:-false}"
US_MICROSTRUCTURE_RECOVER_COLLECT_START="${US_MICROSTRUCTURE_RECOVER_COLLECT_START:-21:25}"
US_MICROSTRUCTURE_RECOVER_COLLECT_END="${US_MICROSTRUCTURE_RECOVER_COLLECT_END:-05:15}"
US_MICROSTRUCTURE_RECOVER_REPORT_START="${US_MICROSTRUCTURE_RECOVER_REPORT_START:-08:30}"
US_MICROSTRUCTURE_RECOVER_REPORT_END="${US_MICROSTRUCTURE_RECOVER_REPORT_END:-18:00}"
US_MICROSTRUCTURE_RECOVER_MIN_COLLECT_SECONDS="${US_MICROSTRUCTURE_RECOVER_MIN_COLLECT_SECONDS:-900}"
US_MICROSTRUCTURE_RECOVER_REQUIRE_EMAIL="${US_MICROSTRUCTURE_RECOVER_REQUIRE_EMAIL:-true}"
US_MICROSTRUCTURE_RECOVER_SEND_EMAIL="${US_MICROSTRUCTURE_RECOVER_SEND_EMAIL:-true}"
LOCK_DIR="${LOCK_DIR:-$PROJECT_DIR/logs/us_microstructure_recover.lock}"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] run_us_microstructure_recover: $*"
}

mkdir -p "$PROJECT_DIR/logs"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
    log "skip (lock exists: $LOCK_DIR)"
    exit 0
fi
trap 'rmdir "$LOCK_DIR"' EXIT

if [ ! -x "$PYTHON_BIN" ]; then
    PYTHON_BIN="$(command -v python3 || true)"
fi
if [ -z "$PYTHON_BIN" ]; then
    log "python not found"
    exit 2
fi

decision="$(
    DATA_DIR="$DATA_DIR" \
    US_MICROSTRUCTURE_DIR="$US_MICROSTRUCTURE_DIR" \
    US_MICROSTRUCTURE_RECOVER_NOW="$US_MICROSTRUCTURE_RECOVER_NOW" \
    US_MICROSTRUCTURE_RECOVER_COLLECT_START="$US_MICROSTRUCTURE_RECOVER_COLLECT_START" \
    US_MICROSTRUCTURE_RECOVER_COLLECT_END="$US_MICROSTRUCTURE_RECOVER_COLLECT_END" \
    US_MICROSTRUCTURE_RECOVER_REPORT_START="$US_MICROSTRUCTURE_RECOVER_REPORT_START" \
    US_MICROSTRUCTURE_RECOVER_REPORT_END="$US_MICROSTRUCTURE_RECOVER_REPORT_END" \
    US_MICROSTRUCTURE_RECOVER_MIN_COLLECT_SECONDS="$US_MICROSTRUCTURE_RECOVER_MIN_COLLECT_SECONDS" \
    US_MICROSTRUCTURE_RECOVER_REQUIRE_EMAIL="$US_MICROSTRUCTURE_RECOVER_REQUIRE_EMAIL" \
    "$PYTHON_BIN" - <<'PY'
import json
import os
from datetime import datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo


def parse_hhmm(value: str) -> time:
    hour, minute = str(value).split(":", 1)
    return time(int(hour), int(minute))


def parse_now(value: str) -> datetime:
    tz = ZoneInfo("Asia/Shanghai")
    if not value:
        return datetime.now(tz)
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=tz)
    return parsed.astimezone(tz)


def collection_dates(base: Path) -> list[str]:
    dates: set[str] = set()
    for kind in ("manifests", "trades", "order_book", "quotes"):
        root = base / kind
        if not root.exists():
            continue
        for path in root.glob("date=*"):
            value = path.name.split("=", 1)[1][:10]
            try:
                datetime.strptime(value, "%Y-%m-%d")
            except ValueError:
                continue
            dates.add(value)
    return sorted(dates)


def report_needed(base: Path, *, require_email: bool) -> tuple[bool, str, str]:
    dates = collection_dates(base)
    report_date = dates[-1] if dates else (now.date() - timedelta(days=1)).isoformat()
    status_path = base / "reports" / f"date={report_date}" / "status.json"
    if not status_path.exists():
        return True, report_date, "missing status"
    try:
        payload = json.loads(status_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return True, report_date, f"status unreadable: {exc}"
    if not bool(payload.get("is_final_report", True)):
        return True, report_date, "latest status is not final"
    if require_email:
        delivery = payload.get("email_delivery") if isinstance(payload, dict) else {}
        if not isinstance(delivery, dict) or not (delivery.get("requested") and delivery.get("sent")):
            return True, report_date, "final report email not sent"
    return False, report_date, "already complete"


now = parse_now(os.environ.get("US_MICROSTRUCTURE_RECOVER_NOW", ""))
collect_start = parse_hhmm(os.environ.get("US_MICROSTRUCTURE_RECOVER_COLLECT_START", "21:25"))
collect_end = parse_hhmm(os.environ.get("US_MICROSTRUCTURE_RECOVER_COLLECT_END", "05:15"))
report_start = parse_hhmm(os.environ.get("US_MICROSTRUCTURE_RECOVER_REPORT_START", "08:30"))
report_end = parse_hhmm(os.environ.get("US_MICROSTRUCTURE_RECOVER_REPORT_END", "18:00"))
min_collect_seconds = int(os.environ.get("US_MICROSTRUCTURE_RECOVER_MIN_COLLECT_SECONDS", "900"))
require_email = os.environ.get("US_MICROSTRUCTURE_RECOVER_REQUIRE_EMAIL", "true").lower() in {"1", "true", "yes", "y"}
base = Path(os.environ.get("US_MICROSTRUCTURE_DIR") or os.environ.get("DATA_DIR", str(Path.home() / "quantpilot_data")) + "/us_microstructure").expanduser()

weekday = now.weekday()
today = now.date()
collect_active = False
collect_end_dt = None
if weekday in range(0, 5) and now.time() >= collect_start:
    collect_active = True
    collect_end_dt = datetime.combine(today + timedelta(days=1), collect_end, tzinfo=now.tzinfo)
elif weekday in range(1, 6) and now.time() < collect_end:
    collect_active = True
    collect_end_dt = datetime.combine(today, collect_end, tzinfo=now.tzinfo)

collect_seconds = 0
collect_reason = "outside collect window"
if collect_active and collect_end_dt is not None:
    collect_seconds = max(0, int((collect_end_dt - now).total_seconds()))
    if collect_seconds >= min_collect_seconds:
        collect_reason = "inside collect window"
    else:
        collect_active = False
        collect_reason = f"remaining collect window too short: {collect_seconds}s"

report_active = weekday in range(1, 6) and report_start <= now.time() <= report_end
needed, report_date, report_reason = report_needed(base, require_email=require_email)
run_report = bool(report_active and needed)
if not report_active:
    report_reason = "outside report recovery window"

print(
    json.dumps(
        {
            "now": now.isoformat(timespec="seconds"),
            "base_dir": str(base),
            "run_collect": bool(collect_active),
            "collect_seconds": int(collect_seconds),
            "collect_reason": collect_reason,
            "run_report": run_report,
            "report_date": report_date,
            "report_reason": report_reason,
        },
        ensure_ascii=False,
    )
)
PY
)"

log "decision=$decision"

run_collect="$(printf '%s' "$decision" | "$PYTHON_BIN" -c 'import json,sys; print(str(json.load(sys.stdin)["run_collect"]).lower())')"
collect_seconds="$(printf '%s' "$decision" | "$PYTHON_BIN" -c 'import json,sys; print(int(json.load(sys.stdin)["collect_seconds"]))')"
run_report="$(printf '%s' "$decision" | "$PYTHON_BIN" -c 'import json,sys; print(str(json.load(sys.stdin)["run_report"]).lower())')"

if [ "$US_MICROSTRUCTURE_RECOVER_DRY_RUN" = "true" ]; then
    if [ "$run_collect" = "true" ]; then
        log "dry-run collect duration=${collect_seconds}s"
    fi
    if [ "$run_report" = "true" ]; then
        log "dry-run report"
    fi
    exit 0
fi

if [ "$run_collect" = "true" ]; then
    log "recover collect duration=${collect_seconds}s"
    US_MICROSTRUCTURE_COLLECT_DURATION_SECONDS="$collect_seconds" \
        /bin/bash "$PROJECT_DIR/scripts/run_us_microstructure_auto.sh" collect
fi

if [ "$run_report" = "true" ]; then
    log "recover report"
    US_MICROSTRUCTURE_SEND_EMAIL="$US_MICROSTRUCTURE_RECOVER_SEND_EMAIL" \
        /bin/bash "$PROJECT_DIR/scripts/run_us_microstructure_auto.sh" report
fi

log "done"
