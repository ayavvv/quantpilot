#!/bin/bash
# Install the US microstructure collection and report launchd daemons.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
INSTALL_USER="${SUDO_USER:-$(id -un)}"
INSTALL_UID="$(id -u "$INSTALL_USER")"
HOME_DIR="$(dscl . -read "/Users/$INSTALL_USER" NFSHomeDirectory | awk '{print $2}')"
TARGET_DIR="/Library/LaunchDaemons"
AGENT_TARGET_DIR="$HOME_DIR/Library/LaunchAgents"
LOG_DIR="$PROJECT_DIR/logs"
INSTALL_MODE="${US_MICROSTRUCTURE_LAUNCHD_MODE:-auto}"

if [ "$INSTALL_MODE" = "auto" ]; then
    if sudo -n true >/dev/null 2>&1; then
        INSTALL_MODE="daemon"
    else
        INSTALL_MODE="agent"
    fi
fi

render_and_install() {
    local label="$1"
    local template_path="$PROJECT_DIR/deploy/launchd/${label}.plist"
    local target_dir="$TARGET_DIR"
    local target_path
    local tmp_path
    tmp_path="$(mktemp "/tmp/${label}.plist.XXXXXX")"

    if [ "$INSTALL_MODE" = "agent" ]; then
        target_dir="$AGENT_TARGET_DIR"
    fi
    target_path="$target_dir/${label}.plist"

    python3 - <<'PY' "$template_path" "$tmp_path" "$PROJECT_DIR" "$HOME_DIR" "$INSTALL_USER" "$INSTALL_MODE"
import plistlib
from pathlib import Path
import sys

template_path = Path(sys.argv[1])
target_path = Path(sys.argv[2])
project_dir = sys.argv[3]
home_dir = sys.argv[4]
user_name = sys.argv[5]
install_mode = sys.argv[6]
content = template_path.read_text(encoding="utf-8")
content = content.replace("__PROJECT_DIR__", project_dir)
content = content.replace("__HOME__", home_dir)
content = content.replace("__USER__", user_name)
payload = plistlib.loads(content.encode("utf-8"))
if install_mode == "agent":
    payload.pop("UserName", None)
with target_path.open("wb") as handle:
    plistlib.dump(payload, handle, sort_keys=False)
PY

    mkdir -p "$target_dir"
    if [ "$INSTALL_MODE" = "daemon" ]; then
        launchctl bootout "gui/$INSTALL_UID/$label" >/dev/null 2>&1 || true
        launchctl bootout "user/$INSTALL_UID/$label" >/dev/null 2>&1 || true
        sudo launchctl bootout "system/$label" >/dev/null 2>&1 || true
        sudo install -m 644 -o root -g wheel "$tmp_path" "$target_path"
        sudo launchctl bootstrap system "$target_path"
        sudo launchctl enable "system/$label"
    elif [ "$INSTALL_MODE" = "agent" ]; then
        launchctl bootout "gui/$INSTALL_UID/$label" >/dev/null 2>&1 || true
        install -m 644 "$tmp_path" "$target_path"
        launchctl bootstrap "gui/$INSTALL_UID" "$target_path"
        launchctl enable "gui/$INSTALL_UID/$label"
    else
        echo "unsupported US_MICROSTRUCTURE_LAUNCHD_MODE: $INSTALL_MODE" >&2
        rm -f "$tmp_path"
        exit 2
    fi
    rm -f "$tmp_path"
    printf 'Installed launchd %s: %s\n' "$INSTALL_MODE" "$target_path"
}

mkdir -p "$LOG_DIR"
render_and_install "com.quantpilot.us_microstructure.collect"
render_and_install "com.quantpilot.us_microstructure.report"

printf 'US microstructure launchd %s jobs installed for user %s\n' "$INSTALL_MODE" "$INSTALL_USER"
printf 'Logs: %s\n' "$LOG_DIR"
