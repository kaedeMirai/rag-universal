import importlib.util
from pathlib import Path


ROOT_SETTINGS_PATH = Path(__file__).resolve().parents[1] / "settings.py"
SPEC = importlib.util.spec_from_file_location("project_root_settings", ROOT_SETTINGS_PATH)

if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to load root settings module from {ROOT_SETTINGS_PATH}")

MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

Settings = MODULE.Settings
load_settings = MODULE.load_settings
settings = MODULE.settings
