#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python src/train.py "$@"
