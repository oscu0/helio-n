from pathlib import Path

from Library.SW.CH_SW_Model import EmpiricalCHSWModel

MODEL = EmpiricalCHSWModel.from_fields(
    source_path=Path(__file__).resolve(),
    v_min=350.0,
    a=180.0,
    alpha=0.6,
)


def load():
    return MODEL
