#!/bin/bash
# Install the Polymarket launchd daemon so it starts at system boot.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
TEMPLATE_PATH="$PROJECT_DIR/deploy/launchd/com.quantpilot.polymarket.scheduler.plist"
LABEL="com.quantpilot.polymarket.scheduler"
INSTALL_USER="${SUDO_USER:-$(id -un)}"
INSTALL_UID="$(id -u "$INSTALL_USER")"
HOME_DIR="$(dscl . -read "/Users/$INSTALL_USER" NFSHomeDirectory | awk '{print $2}')"
TARGET_DIR="/Library/LaunchDaemons"
TARGET_PATH="$TARGET_DIR/com.quantpilot.polymarket.scheduler.plist"
OLD_AGENT_PATH="$HOME_DIR/Library/LaunchAgents/com.quantpilot.polymarket.scheduler.plist"
LOG_DIR="$PROJECT_DIR/logs"
TMP_PATH="$(mktemp "/tmp/${LABEL}.XXXXXX.plist")"
trap 'rm -f "$TMP_PATH"' EXIT

mkdir -p "$LOG_DIR"

python3 - <<'PY' "$TEMPLATE_PATH" "$TMP_PATH" "$PROJECT_DIR" "$HOME_DIR" "$INSTALL_USER"
from pathlib import Path
import sys

template_path = Path(sys.argv[1])
target_path = Path(sys.argv[2])
project_dir = sys.argv[3]
home_dir = sys.argv[4]
user_name = sys.argv[5]
content = template_path.read_text(encoding='utf-8')
content = content.replace('__PROJECT_DIR__', project_dir)
content = content.replace('__HOME__', home_dir)
content = content.replace('__USER__', user_name)
target_path.write_text(content, encoding='utf-8')
PY

launchctl bootout "gui/$INSTALL_UID/$LABEL" >/dev/null 2>&1 || true
launchctl bootout "user/$INSTALL_UID/$LABEL" >/dev/null 2>&1 || true
rm -f "$OLD_AGENT_PATH"

sudo launchctl bootout "system/$LABEL" >/dev/null 2>&1 || true
sudo install -m 644 -o root -g wheel "$TMP_PATH" "$TARGET_PATH"
sudo launchctl bootstrap system "$TARGET_PATH"
sudo launchctl enable "system/$LABEL"
sudo launchctl kickstart -k "system/$LABEL"

printf 'Installed launchd daemon at %s for user %s\n' "$TARGET_PATH" "$INSTALL_USER"
printf 'Logs: %s\n' "$LOG_DIR"
