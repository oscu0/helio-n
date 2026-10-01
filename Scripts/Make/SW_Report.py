#!/usr/bin/env python3
"""Compatibility entry point for the unified solar-wind analysis worker."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Scripts.Make.SW_Stats import main


if __name__ == "__main__":
    raise SystemExit(main())
