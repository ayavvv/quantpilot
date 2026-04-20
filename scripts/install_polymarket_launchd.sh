#!/bin/bash
# Install the Polymarket launchd agent for the current user.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
TEMPLATE_PATH="$PROJECT_DIR/deploy/launchd/com.quantpilot.polymarket.scheduler.plist"
TARGET_DIR="$HOME/Library/LaunchAgents"
TARGET_PATH="$TARGET_DIR/com.quantpilot.polymarket.scheduler.plist"
LOG_DIR="$PROJECT_DIR/logs"

mkdir -p "$TARGET_DIR" "$LOG_DIR"

python3 - <<'PY' "$TEMPLATE_PATH" "$TARGET_PATH" "$PROJECT_DIR" "$HOME"
from pathlib import Path
import sys

template_path = Path(sys.argv[1])
target_path = Path(sys.argv[2])
project_dir = sys.argv[3]
home_dir = sys.argv[4]
content = template_path.read_text(encoding='utf-8')
content = content.replace('__PROJECT_DIR__', project_dir)
content = content.replace('__HOME__', home_dir)
target_path.write_text(content, encoding='utf-8')
PY

launchctl bootout "gui/$(id -u)/com.quantpilot.polymarket.scheduler" >/dev/null 2>&1 || true
launchctl bootstrap "gui/$(id -u)" "$TARGET_PATH"
launchctl enable "gui/$(id -u)/com.quantpilot.polymarket.scheduler"
launchctl kickstart -k "gui/$(id -u)/com.quantpilot.polymarket.scheduler"

printf 'Installed launchd agent at %s\n' "$TARGET_PATH"
printf 'Logs: %s\n' "$LOG_DIR"
