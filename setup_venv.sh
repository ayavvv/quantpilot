#!/usr/bin/env bash
# 使用 Python 3.12 创建虚拟环境并安装全部依赖（含 pyqlib）
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

PY="${PYTHON:-python3.12}"
if ! command -v "$PY" &>/dev/null; then
  echo "未找到 $PY，请先安装 Python 3.12："
  echo "  brew install python@3.12   # macOS"
  echo "  或 pyenv install 3.12 && pyenv local 3.12"
  exit 1
fi

"$PY" --version
rm -rf .venv
"$PY" -m venv .venv
# shellcheck disable=SC1091
. .venv/bin/activate
pip install -U pip
pip install -r requirements.txt -r requirements-qlib.txt
echo "环境已就绪，激活命令: source .venv/bin/activate"
echo "运行全流程: python main.py"
