import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(dotenv_path='C:\\Users\\shayl\\OneDrive - NTNU\\edac_repo\\edac_work\\.env')

def _fix_path(env_name: str) -> Path:
    _dir = os.getenv(env_name)
    if not _dir:
        raise OSError(f"Please update {env_name} in .env")
    _path = Path(_dir)
    if not _path.exists():
        raise FileNotFoundError(f"{_dir} do not exist")
    return _path


def get_project_root() -> Path:
    return _fix_path("LOCAL_WORKING_DIR")


def get_data_dir() -> Path:
    return _fix_path("DATA_DIR")
