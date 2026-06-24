from pathlib import Path
import shutil
import sys

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def pytest_sessionfinish(session, exitstatus):
    shutil.rmtree(ROOT / "tests" / "__pycache__", ignore_errors=True)
    shutil.rmtree(ROOT / "python_controller" / "__pycache__", ignore_errors=True)
