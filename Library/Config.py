from pathlib import Path
import json
import socket
import os

hostname = socket.gethostname()
override_machine = os.environ.get("MACHINE")

SCRIPT_DIR = Path(__file__).resolve().parent


with open(SCRIPT_DIR / "../Config/Machine.json", "r") as f:
    machines = json.load(f)
    key = override_machine if override_machine else hostname
    machine_config = machines.get(key, next(iter(machines.values())))

paths = {
    "fits_root": machine_config["fits_root"],
    "masks_root": machine_config["masks_root"],
    "hmi_root": machine_config["hmi_root"],
    "artifact_root": machine_config["artifact_root"],
}

# Ensure artifact_root exists
Path(paths["artifact_root"]).mkdir(parents=True, exist_ok=True)

apply_config = {
    "batch_size": machine_config["apply_batch_size"],
    "max_inflight_plots": machine_config["max_inflight_plots"],
    "plot_threads": machine_config["plot_threads"],
    "chunk_size": machine_config["chunk_size"]
}

train_batch_size = int(machine_config["train_batch_size"])

with open(SCRIPT_DIR / "../Config/Plot.json", "r") as f:
    plot_config = json.load(f)

TARGET_PX = int(plot_config["target_px"])
DPI = int(plot_config["dpi"])
