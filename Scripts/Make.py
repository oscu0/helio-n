#!/usr/bin/env python3
import sys
import subprocess
from pathlib import Path


def main(argv):
    scripts_dir = Path(__file__).resolve().parent / "Make"

    available = {}
    for path in scripts_dir.glob("*.py"):
        if path.name.startswith("__"):
            continue
        available[path.stem.lower()] = path

    if not available:
        print(f"No Make scripts found in {scripts_dir}")
        return 1

    if len(argv) == 1:
        names = sorted(p.stem for p in available.values())
        print("Available Make scripts:")
        for name in names:
            print(f"- {name}")
        return 0

    target = argv[1].lower()
    rest = argv[2:]

    script = available.get(target)
    if script is None:
        names = ", ".join(sorted(p.stem for p in available.values()))
        print(f"Unknown Make script: {argv[1]}")
        print(f"Available: {names}")
        return 1

    cmd = [sys.executable, str(script), *rest]
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
