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

TARGET_PX = 1024
DPI = 128
