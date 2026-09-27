import json
import os
import shutil
import threading
from pathlib import Path

import pytest

from packing.threejs_visualization.frame import SceneFrame, VisualizationFrame
from packing.threejs_visualization.replay import (
    ReplayPublishRecoveryError,
    ThreeReplayRecorder,
    normalize_three_replay_path,
)


def _frame(title: str) -> VisualizationFrame:
    empty = SceneFrame(bounds=[0, 0, 0, 1, 1, 1], objects=[])
    return VisualizationFrame(
        title=title,
        scenes={"container": empty, "buffer": empty, "holding": empty},
        legend=[],
    )


@pytest.fixture
def runtime_assets(tmp_path: Path) -> Path:
    static = tmp_path / "runtime"
    (static / "licenses").mkdir(parents=True)
    (static / "live.html").write_text("live-only", encoding="utf-8")
    (static / "replay.html").write_text(
        """<!doctype html><link rel="stylesheet" href="static/style.css">
<script src="frames.js"></script>
<script src="static/renderer.bundle.js"></script>""",
        encoding="utf-8",
    )
    (static / "style.css").write_text("body { color: black; }", encoding="utf-8")
    (static / "renderer.bundle.js").write_text(
        "globalThis.PackingThree = {};", encoding="utf-8"
    )
    (static / "licenses" / "three-LICENSE.txt").write_text(
        "MIT License", encoding="utf-8"
    )
    return static


def _decode_frames_js(path: Path) -> dict:
    prefix = "window.PACKING_THREE_REPLAY = "
    source = path.read_text(encoding="utf-8")
    assert source.startswith(prefix)
    assert source.endswith(";\n")
    return json.loads(source[len(prefix) : -2])


def test_normalize_html_replay_path_to_directory():
    assert normalize_three_replay_path("results/demo.html") == Path("results/demo")
    assert normalize_three_replay_path("results/demo.HTML") == Path("results/demo")
    assert normalize_three_replay_path(Path("results/demo")) == Path("results/demo")


def test_save_creates_minimal_movable_offline_bundle(tmp_path, runtime_assets):
    out_dir = tmp_path / "demo"
    recorder = ThreeReplayRecorder(out_dir, interval_ms=700, assets_root=runtime_assets)
    recorder.capture("first", _frame("frame one"))
    recorder.capture("second", {"title": "frame two", "scenes": {}, "legend": []})

    index_path = recorder.save()

    assert index_path == out_dir / "index.html"
    assert (out_dir / "index.html").is_file()
    assert (out_dir / "frames.js").is_file()
    assert (out_dir / "static" / "renderer.bundle.js").is_file()
    assert (out_dir / "static" / "style.css").is_file()
    assert (out_dir / "static" / "licenses" / "three-LICENSE.txt").is_file()
    assert not (out_dir / "static" / "live.html").exists()
    assert not (out_dir / "static" / "replay.html").exists()

    index = (out_dir / "index.html").read_text(encoding="utf-8")
    assert "fetch(" not in index
    assert 'type="module"' not in index
    assert 'src="frames.js"' in index
    assert 'src="static/renderer.bundle.js"' in index
    assert 'href="static/style.css"' in index
    all_javascript = "\n".join(
        path.read_text(encoding="utf-8") for path in out_dir.rglob("*.js")
    )
    assert "https://" not in all_javascript
    assert "http://" not in all_javascript

    replay = _decode_frames_js(out_dir / "frames.js")
    assert replay["interval_ms"] == 700
    assert [entry["title"] for entry in replay["frames"]] == ["first", "second"]
    assert replay["frames"][0]["frame"]["title"] == "frame one"


def test_capture_validates_and_snapshots_input(tmp_path, runtime_assets):
    recorder = ThreeReplayRecorder(tmp_path / "demo", assets_root=runtime_assets)
    payload = {"title": "before", "nested": {"items": [1]}}

    recorder.capture("safe </script> title", payload)
    payload["title"] = "after"
    payload["nested"]["items"].append(2)
    recorder.save()

    saved = _decode_frames_js(tmp_path / "demo" / "frames.js")
    assert saved["frames"] == [
        {
            "title": "safe </script> title",
            "frame": {"title": "before", "nested": {"items": [1]}},
        }
    ]


@pytest.mark.parametrize("title", [None, 3, ""])
def test_capture_rejects_invalid_titles(tmp_path, runtime_assets, title):
    recorder = ThreeReplayRecorder(tmp_path / "demo", assets_root=runtime_assets)
    with pytest.raises((TypeError, ValueError), match="title"):
        recorder.capture(title, {})


@pytest.mark.parametrize("value", [None, 3, "frame", [], object()])
def test_capture_accepts_only_visualization_frames_or_dicts(
    tmp_path, runtime_assets, value
):
    recorder = ThreeReplayRecorder(tmp_path / "demo", assets_root=runtime_assets)
    with pytest.raises(TypeError, match="VisualizationFrame or dict"):
        recorder.capture("frame", value)


def test_capture_rejects_invalid_json_without_adding_frame(tmp_path, runtime_assets):
    recorder = ThreeReplayRecorder(tmp_path / "demo", assets_root=runtime_assets)
    with pytest.raises(TypeError, match="JSON serializable"):
        recorder.capture("bad", {"value": object()})
    with pytest.raises(ValueError, match="finite"):
        recorder.capture("bad", {"value": float("nan")})

    recorder.capture("valid", {})
    recorder.save()
    assert [entry["title"] for entry in _decode_frames_js(
        tmp_path / "demo" / "frames.js"
    )["frames"]] == ["valid"]


@pytest.mark.parametrize("interval", [True, 0, -1, 1.5, "700", None])
def test_rejects_invalid_interval(tmp_path, runtime_assets, interval):
    with pytest.raises((TypeError, ValueError), match="interval_ms"):
        ThreeReplayRecorder(
            tmp_path / "demo", interval_ms=interval, assets_root=runtime_assets
        )


def test_repeated_save_updates_owned_files_and_preserves_unrelated_files(
    tmp_path, runtime_assets
):
    out_dir = tmp_path / "demo"
    recorder = ThreeReplayRecorder(out_dir, assets_root=runtime_assets)
    recorder.capture("first", {})
    recorder.save()
    unrelated = out_dir / "notes.txt"
    unrelated.write_text("keep me", encoding="utf-8")

    recorder.capture("second", {})
    assert recorder.save() == out_dir / "index.html"

    assert unrelated.read_text(encoding="utf-8") == "keep me"
    assert [entry["title"] for entry in _decode_frames_js(
        out_dir / "frames.js"
    )["frames"]] == ["first", "second"]


def test_save_validates_assets_before_creating_output(tmp_path):
    missing_assets = tmp_path / "missing-assets"
    out_dir = tmp_path / "demo"
    recorder = ThreeReplayRecorder(out_dir, assets_root=missing_assets)

    with pytest.raises(FileNotFoundError, match="live.html"):
        recorder.save()

    assert not out_dir.exists()


def test_staging_failure_leaves_existing_bundle_untouched_and_cleans_temp_dirs(
    tmp_path, runtime_assets, monkeypatch
):
    out_dir = tmp_path / "demo"
    recorder = ThreeReplayRecorder(out_dir, assets_root=runtime_assets)
    recorder.capture("old", {})
    recorder.save()
    owned_before = {
        path.relative_to(out_dir): path.read_bytes()
        for path in out_dir.rglob("*")
        if path.is_file()
    }

    recorder.capture("new", {})
    (runtime_assets / "replay.html").write_text(
        "new index that must not be published", encoding="utf-8"
    )
    real_copy2 = shutil.copy2
    calls = 0

    def fail_second_copy(source, destination, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected staging failure")
        return real_copy2(source, destination, *args, **kwargs)

    monkeypatch.setattr(shutil, "copy2", fail_second_copy)
    with pytest.raises(OSError, match="injected staging failure"):
        recorder.save()

    owned_after = {
        path.relative_to(out_dir): path.read_bytes()
        for path in out_dir.rglob("*")
        if path.is_file()
    }
    assert owned_after == owned_before
    assert not list(tmp_path.glob(".demo.*-*"))


def test_publish_failure_rolls_back_owned_files_and_preserves_unrelated_files(
    tmp_path, runtime_assets, monkeypatch
):
    out_dir = tmp_path / "demo"
    recorder = ThreeReplayRecorder(out_dir, assets_root=runtime_assets)
    recorder.capture("old", {})
    recorder.save()
    unrelated = out_dir / "notes.txt"
    unrelated.write_text("keep me", encoding="utf-8")
    owned_paths = (
        Path("index.html"),
        Path("frames.js"),
        Path("static/renderer.bundle.js"),
        Path("static/style.css"),
        Path("static/licenses/three-LICENSE.txt"),
    )
    owned_before = {
        relative: (out_dir / relative).read_bytes() for relative in owned_paths
    }

    recorder.capture("new", {})
    real_replace = os.replace
    failed = False

    def fail_frames_publish(source, destination):
        nonlocal failed
        source_path = Path(source)
        destination_path = Path(destination)
        if (
            not failed
            and source_path.name == "frames.js"
            and destination_path == out_dir / "frames.js"
        ):
            failed = True
            raise OSError("injected publish failure")
        return real_replace(source, destination)

    monkeypatch.setattr(os, "replace", fail_frames_publish)
    with pytest.raises(OSError, match="injected publish failure"):
        recorder.save()

    assert failed
    assert unrelated.read_text(encoding="utf-8") == "keep me"
    assert {
        relative: (out_dir / relative).read_bytes() for relative in owned_paths
    } == owned_before
    assert not list(tmp_path.glob(".demo.*-*"))


def test_publish_failure_removes_owned_files_created_by_failed_save(
    tmp_path, runtime_assets, monkeypatch
):
    out_dir = tmp_path / "demo"
    out_dir.mkdir()
    unrelated = out_dir / "notes.txt"
    unrelated.write_text("keep me", encoding="utf-8")
    recorder = ThreeReplayRecorder(out_dir, assets_root=runtime_assets)
    recorder.capture("new", {})
    real_replace = os.replace
    failed = False

    def fail_frames_publish(source, destination):
        nonlocal failed
        if not failed and Path(destination) == out_dir / "frames.js":
            failed = True
            raise OSError("injected publish failure")
        return real_replace(source, destination)

    monkeypatch.setattr(os, "replace", fail_frames_publish)
    with pytest.raises(OSError, match="injected publish failure"):
        recorder.save()

    assert unrelated.read_text(encoding="utf-8") == "keep me"
    assert not (out_dir / "index.html").exists()
    assert not (out_dir / "frames.js").exists()
    assert not (out_dir / "static" / "renderer.bundle.js").exists()
    assert not list(tmp_path.glob(".demo.*-*"))


def test_rollback_failure_restores_other_files_and_retains_recovery_backup(
    tmp_path, runtime_assets, monkeypatch
):
    out_dir = tmp_path / "demo"
    recorder = ThreeReplayRecorder(out_dir, assets_root=runtime_assets)
    recorder.capture("old", {})
    recorder.save()
    unrelated = out_dir / "notes.txt"
    unrelated.write_text("keep me", encoding="utf-8")
    owned_paths = (
        Path("index.html"),
        Path("frames.js"),
        Path("static/renderer.bundle.js"),
        Path("static/style.css"),
        Path("static/licenses/three-LICENSE.txt"),
    )
    owned_before = {
        relative: (out_dir / relative).read_bytes() for relative in owned_paths
    }
    recorder.capture("new", {})
    real_replace = os.replace
    publish_failed = False
    restore_failed = False

    def fail_publish_and_one_restore(source, destination):
        nonlocal publish_failed, restore_failed
        source_path = Path(source)
        destination_path = Path(destination)
        if (
            not publish_failed
            and source_path.name == "three-LICENSE.txt"
            and destination_path
            == out_dir / "static/licenses/three-LICENSE.txt"
            and "bundle" in source_path.parts
        ):
            publish_failed = True
            raise OSError("injected publish failure")
        if (
            not restore_failed
            and source_path.name == "renderer.bundle.js"
            and destination_path == out_dir / "static/renderer.bundle.js"
            and "backup" in source_path.parts
        ):
            restore_failed = True
            raise OSError("injected restore failure")
        return real_replace(source, destination)

    monkeypatch.setattr(os, "replace", fail_publish_and_one_restore)
    with pytest.raises(
        ReplayPublishRecoveryError, match="injected publish failure"
    ) as error:
        recorder.save()

    assert publish_failed and restore_failed
    assert "recovery" in str(error.value).lower()
    assert isinstance(error.value.original_error, OSError)
    assert str(error.value.original_error) == "injected publish failure"
    assert len(error.value.rollback_errors) == 1
    assert isinstance(error.value.rollback_errors[0], OSError)
    assert str(error.value.rollback_errors[0]) == "injected restore failure"
    assert error.value.__cause__ is error.value.original_error
    recovery_dir = Path(error.value.recovery_dir)
    assert recovery_dir.is_dir()
    assert recovery_dir.name.startswith(".demo.recovery-")
    assert error.value.recovery_dir == recovery_dir
    failed_relative = Path("static/renderer.bundle.js")
    assert (recovery_dir / "backup" / failed_relative).read_bytes() == owned_before[
        failed_relative
    ]
    assert not (out_dir / failed_relative).exists()
    # frames.js and index.html occur *after* renderer.bundle.js in the reverse
    # rollback order. Their restoration proves one failure did not stop it.
    for relative in owned_paths:
        if relative == failed_relative:
            continue
        assert (out_dir / relative).read_bytes() == owned_before[relative]
    assert unrelated.read_text(encoding="utf-8") == "keep me"
    assert not list(tmp_path.glob(".demo.stage-*"))


def test_concurrent_saves_on_one_recorder_serialize_publication(
    tmp_path, runtime_assets, monkeypatch
):
    recorder = ThreeReplayRecorder(tmp_path / "demo", assets_root=runtime_assets)
    recorder.capture("frame", {})
    attempts = {
        "first": threading.Event(),
        "second": threading.Event(),
    }
    acquired = {
        "first": threading.Event(),
        "second": threading.Event(),
    }
    real_lock = threading.Lock()

    class ObservableLock:
        def __enter__(self):
            name = threading.current_thread().name
            attempts[name].set()
            real_lock.acquire()
            acquired[name].set()
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            real_lock.release()

    recorder._save_lock = ObservableLock()
    real_publish = recorder._publish_bundle
    first_in_publication = threading.Event()
    allow_first_to_finish = threading.Event()
    publication_order = []
    errors = []

    def observed_publish(staged, backup):
        name = threading.current_thread().name
        publication_order.append(f"enter:{name}")
        if name == "first":
            first_in_publication.set()
            if not allow_first_to_finish.wait(timeout=2):
                raise TimeoutError("test did not release first publication")
        real_publish(staged, backup)
        publication_order.append(f"exit:{name}")

    monkeypatch.setattr(recorder, "_publish_bundle", observed_publish)

    def save_in_thread():
        try:
            recorder.save()
        except BaseException as error:
            errors.append(error)

    first = threading.Thread(target=save_in_thread, name="first")
    second = threading.Thread(target=save_in_thread, name="second")
    first.start()
    assert first_in_publication.wait(timeout=2)
    second.start()
    assert attempts["second"].wait(timeout=2)
    assert not acquired["second"].is_set()
    assert publication_order == ["enter:first"]

    allow_first_to_finish.set()
    first.join(timeout=2)
    second.join(timeout=2)

    assert not first.is_alive() and not second.is_alive()
    assert errors == []
    assert acquired["second"].is_set()
    assert publication_order == [
        "enter:first",
        "exit:first",
        "enter:second",
        "exit:second",
    ]


@pytest.mark.parametrize("symlink_relative", [Path("static"), Path("static/licenses")])
def test_save_rejects_symlink_output_directories(
    tmp_path, runtime_assets, symlink_relative
):
    out_dir = tmp_path / "demo"
    out_dir.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    link = out_dir / symlink_relative
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(external, target_is_directory=True)
    recorder = ThreeReplayRecorder(out_dir, assets_root=runtime_assets)
    recorder.capture("frame", {})

    with pytest.raises(ValueError, match="symbolic link"):
        recorder.save()

    assert link.is_symlink()
    assert list(external.iterdir()) == []
    assert not list(tmp_path.glob(".demo.*-*"))


@pytest.mark.parametrize(
    ("relative", "broken"),
    [(Path("index.html"), False), (Path("frames.js"), True)],
)
def test_save_rejects_owned_destination_symlinks(
    tmp_path, runtime_assets, relative, broken
):
    out_dir = tmp_path / "demo"
    out_dir.mkdir()
    external = tmp_path / "external.txt"
    if not broken:
        external.write_text("do not overwrite", encoding="utf-8")
    link = out_dir / relative
    link.symlink_to(external)
    recorder = ThreeReplayRecorder(out_dir, assets_root=runtime_assets)
    recorder.capture("frame", {})

    with pytest.raises(ValueError, match="symbolic link"):
        recorder.save()

    assert link.is_symlink()
    if not broken:
        assert external.read_text(encoding="utf-8") == "do not overwrite"
    assert not list(tmp_path.glob(".demo.*-*"))
