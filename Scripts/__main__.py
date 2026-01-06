#!/usr/bin/env python3
from pathlib import Path


def main():
    scripts_dir = Path(__file__).resolve().parent
    top_level = []
    for path in scripts_dir.glob("*.py"):
        if path.name.startswith("__"):
            continue
        top_level.append(path.stem)

    make_dir = scripts_dir / "Make"
    make_scripts = []
    if make_dir.exists():
        for path in make_dir.glob("*.py"):
            if path.name.startswith("__"):
                continue
            make_scripts.append(path.stem)

    top_level = sorted(top_level, key=str.lower)
    make_scripts = sorted(make_scripts, key=str.lower)

    print("Available commands:")
    for name in top_level:
        if name.lower() == "make" and make_scripts:
            print("- Make")
            for sub in make_scripts:
                print(f"  - {sub}")
        else:
            print(f"- {name}")


if __name__ == "__main__":
    main()
