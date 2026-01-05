import json
import subprocess
import sys
import socket
from pathlib import Path

import Library.IO as IO
import pandas as pd

BASE_DIR = str(Path(__file__).resolve().parent.parent) + "/"


def build_move_list(src_df, dst_df, kind):
    """Return (to_copy, missing_sources, already_present_count) for a path column."""
    col = f"{kind}_path"
    moves = []
    missing = []
    already_present = 0
    seen = set()

    for src, dst in zip(src_df[col], dst_df[col]):
        if pd.isna(src) or pd.isna(dst):
            continue
        src_path = Path(src)
        dst_path = Path(dst)
        pair = (src_path, dst_path)
        if pair in seen or src_path == dst_path:
            continue
        seen.add(pair)
        if not src_path.exists():
            missing.append(src_path)
            continue
        if dst_path.exists():
            already_present += 1
            continue
        moves.append(pair)

    return moves, missing, already_present


def copy_with_rsync(pairs):
    for src_path, dst_path in pairs:
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = ["rsync", "-a", "--ignore-existing", str(src_path), str(dst_path)]
        try:
            subprocess.run(cmd, check=True)
        except FileNotFoundError:
            print("rsync not found on PATH; aborting.")
            sys.exit(1)
        except subprocess.CalledProcessError as e:
            print(f"rsync failed for {src_path} -> {dst_path}: {e}")


def copy_root_tree(src_root, dst_root):
    """Rsync an entire directory tree from src_root into dst_root."""
    src_root = Path(src_root)
    dst_root = Path(dst_root)
    if not src_root.exists():
        print(f"Source root missing, skipping: {src_root}")
        return
    dst_root.mkdir(parents=True, exist_ok=True)
    cmd = ["rsync", "-a", "--ignore-existing", f"{src_root}/", f"{dst_root}/"]
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        print("rsync not found on PATH; aborting.")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"rsync failed for {src_root} -> {dst_root}: {e}")


def main():
    try:
        direction = sys.argv[1].lower()
    except IndexError:
        print('Pass "up", "down", or "inplace" as the first argument.')
        sys.exit(1)

    if direction not in {"up", "down", "inplace"}:
        print('Direction must be "up", "down", or "inplace".')
        sys.exit(1)

    configs = json.load(open(BASE_DIR + "Config/Machine.json"))

    host = socket.gethostname()
    artifact_root = configs.get(host, configs.get(list(configs.keys())[0])).get(
        "artifact_root", f"./Outputs/Artifacts/{host}/"
    )

    # Only miracle <-> miracle_mini make sense for copy modes
    roots = {
        "fits": [configs["miracle"]["fits_root"], configs["miracle_mini"]["fits_root"]],
        "mask": [configs["miracle"]["masks_root"], configs["miracle_mini"]["masks_root"]],
        "hmi": [configs["miracle"]["hmi_root"], configs["miracle_mini"]["hmi_root"]],
    }

    if direction == "inplace":
        df = IO.synoptic_dataset(
            pd.read_parquet(Path(artifact_root) / "Paths.parquet")
        )
        return

    root_summary = {
        k: {
            "src": v[0] if direction == "up" else v[1],
            "dst": v[1] if direction == "up" else v[0],
        }
        for k, v in roots.items()
    }
    print(f"Copy direction: {direction}")
    print("Roots:", root_summary)

    if direction == "down":
        print("Down mode: rsyncing entire new roots into old roots (ignore existing).")
        for kind, paths in root_summary.items():
            print(f"{kind}: {paths['src']} -> {paths['dst']}")
            copy_root_tree(paths["src"], paths["dst"])
            return

    df = IO.synoptic_dataset(
        pd.read_parquet(Path(artifact_root) / "Paths.parquet")
    )

    df_new = df.copy()
    for kind in ("fits", "mask", "hmi"):
        df_new[f"{kind}_path"] = df_new[f"{kind}_path"].str.replace(
            roots[kind][0], roots[kind][1], regex=False
        )

    # "up": copy miracle -> miracle_mini
    src_df, dst_df = df, df_new

    move_plan = {}
    missing_plan = {}
    for kind in ("fits", "mask", "hmi"):
        moves, missing, already_present = build_move_list(src_df, dst_df, kind)
        move_plan[kind] = moves
        if missing:
            missing_plan[kind] = missing
        print(
            f"{kind}: {len(moves)} to copy, {already_present} already at destination, {len(missing)} missing at source"
        )
        if moves:
            print(f"  example: {moves[0][0]} -> {moves[0][1]}")

    for kind, moves in move_plan.items():
        if not moves:
            continue
        print(f"Copying {len(moves)} {kind} files...")
        copy_with_rsync(moves)

    if missing_plan:
        print("Missing source files:")
        for kind, paths in missing_plan.items():
            print(f"  {kind}: {len(paths)} (e.g., {paths[0]})")


if __name__ == "__main__":
    main()
