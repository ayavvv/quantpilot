#!/bin/bash
# Install the US microstructure collection and report launchd daemons.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
INSTALL_USER="${SUDO_USER:-$(id -un)}"
INSTALL_UID="$(id -u "$INSTALL_USER")"
HOME_DIR="$(dscl . -read "/Users/$INSTALL_USER" NFSHomeDirectory | awk '{print $2}')"
TARGET_DIR="/Library/LaunchDaemons"
LOG_DIR="$PROJECT_DIR/logs"

render_and_install() {
    local label="$1"
    local template_path="$PROJECT_DIR/deploy/launchd/${label}.plist"
    local target_path="$TARGET_DIR/${label}.plist"
    local tmp_path
    tmp_path="$(mktemp "/tmp/${label}.XXXXXX.plist")"

    python3 - <<'PY' "$template_path" "$tmp_path" "$PROJECT_DIR" "$HOME_DIR" "$INSTALL_USER"
from pathlib import Path
import sys

template_path = Path(sys.argv[1])
target_path = Path(sys.argv[2])
project_dir = sys.argv[3]
home_dir = sys.argv[4]
user_name = sys.argv[5]
content = template_path.read_text(encoding="utf-8")
content = content.replace("__PROJECT_DIR__", project_dir)
content = content.replace("__HOME__", home_dir)
content = content.replace("__USER__", user_name)
target_path.write_text(content, encoding="utf-8")
PY

    launchctl bootout "gui/$INSTALL_UID/$label" >/dev/null 2>&1 || true
    launchctl bootout "user/$INSTALL_UID/$label" >/dev/null 2>&1 || true
    sudo launchctl bootout "system/$label" >/dev/null 2>&1 || true
    sudo install -m 644 -o root -g wheel "$tmp_path" "$target_path"
    sudo launchctl bootstrap system "$target_path"
    sudo launchctl enable "system/$label"
    rm -f "$tmp_path"
    printf 'Installed launchd daemon: %s\n' "$target_path"
}

mkdir -p "$LOG_DIR"
render_and_install "com.quantpilot.us_microstructure.collect"
render_and_install "com.quantpilot.us_microstructure.report"

printf 'US microstructure launchd daemons installed for user %s\n' "$INSTALL_USER"
printf 'Logs: %s\n' "$LOG_DIR"
