#!/bin/bash
# Start ScreenPeek. Any arguments are passed straight through, e.g.
#     ./run.sh --preview
#     ./run.sh --min-faces 3 --sound
cd "$(dirname "$0")"

if [ ! -x ./.venv/bin/python ]; then
  echo "No virtual environment found. Run ./install.sh first."
  exit 1
fi

exec ./.venv/bin/python screenpeek.py "$@"
