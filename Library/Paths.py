from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "Data"


def project_path(*parts):
    return PROJECT_ROOT.joinpath(*parts)


def data_path(*parts):
    return DATA_ROOT.joinpath(*parts)


def resolve_repo_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path
