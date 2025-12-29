from pathlib import Path
import json
import socket

hostname = socket.gethostname()

SCRIPT_DIR = Path(__file__).resolve().parent


with open(SCRIPT_DIR / "../Config/Paths.json", "r") as f:
    p = json.load(f)
    if hostname in p:
        paths = p[hostname]
    else:
        paths = p[p.keys()[0]]  # Default to first entry if hostname not found

paths["artifact_root"] = "./Outputs/Artifacts/" + hostname + "/"

with open(SCRIPT_DIR / "../Config/Plot.json", "r") as f:
    p = json.load(f)
    if hostname in p:
        plot_config = p[hostname]
    else:
        plot_config = p[p.keys()[0]]  # Default to first entry if hostname not found

TARGET_PX = int(plot_config.get("target_px", 1024))
DPI = int(plot_config.get("dpi", 128))
