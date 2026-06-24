from pathlib import Path
import re


SCRIPT = Path(__file__).resolve().parents[1] / "run_stability_sweep_strict_convex_compare_3parallel.sh"


def test_launcher_uses_true_headless_by_default():
    text = SCRIPT.read_text()

    assert 'COPPELIASIM_TRUE_HEADLESS="${COPPELIASIM_TRUE_HEADLESS:-1}"' in text
    assert re.search(
        r'\./coppeliaSim\.sh\s+"\$\{headless_args\[@\]\}"\s+-GzmqRemoteApi\.rpcPort="\$\{port\}"',
        text,
    )
