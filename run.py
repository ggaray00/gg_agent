#!/usr/bin/env python3
"""Run gg-agent straight from the source tree, no install needed.

    python run.py "count the python files in this repo"
    python run.py -p anthropic
    python run.py --list-tools
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gg_agent.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
