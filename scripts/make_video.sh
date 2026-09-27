#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
video_python="${FASTGATE_PYTHON:-$project_root/.venv/bin/python}"
if [[ ! -x "$video_python" ]]; then
  video_python="python3"
fi
exec "$video_python" "$project_root/scripts/render_demo.py" "$@"
