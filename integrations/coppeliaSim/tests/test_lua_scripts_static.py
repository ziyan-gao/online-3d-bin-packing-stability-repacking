from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1]


def test_coppeliasim_lua_scripts_do_not_use_pcall():
    offenders = []
    for script in sorted(SCRIPT_DIR.glob("*.lua")):
        text = script.read_text()
        if "pcall" in text:
            offenders.append(script.name)

    assert offenders == []
