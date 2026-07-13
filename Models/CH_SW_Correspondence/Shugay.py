from pathlib import Path

from Library.SW.CH_SW_Model import EmpiricalCHSWModel
from Library.SW.Constants import SW_MODEL_V2_HANDOFF

MODEL = EmpiricalCHSWModel(
    source_path=Path(__file__).resolve(),
    v_min=300.0,
    a=180.0,
    alpha=0.6,
    a_before_handoff=210.0,
    alpha_before_handoff=0.4,
    parameter_handoff=SW_MODEL_V2_HANDOFF,
)


def load():
    return MODEL
