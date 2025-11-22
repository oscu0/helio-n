from pathlib import Path
import json

SCRIPT_DIR = Path(__file__).resolve().parent


with open(SCRIPT_DIR / "../Config/Smoothing Params.json", "r") as f:
    smoothing_params = json.load(f)

with open(SCRIPT_DIR / "../Config/Paths.json", "r") as f:
    paths = json.load(f)

with open(SCRIPT_DIR / "../Config/Training Params.json", "r") as f:
    model_params = json.load(f)