from pathlib import Path
import json

SCRIPT_DIR = Path(__file__).resolve().parent


with open(Path(__file__).resolve().parent / "../Config/Smoothing Params.json", "r") as f:
    smoothing_params = json.load(f)

with open(Path(__file__).resolve().parent / "../Config/Paths.json", "r") as f:
    paths = json.load(f)

with open(Path(__file__).resolve().parent / "../Config/Training Params.json", "r") as f:
    model_params = json.load(f)