import importlib.util
from pathlib import Path


_BASE_DIR = Path(__file__).resolve().parent / "DS-TSK-MIL"


def load_module(module_name):
    module_path = _BASE_DIR / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(f"ds_tsk_mil_{module_name}", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module
