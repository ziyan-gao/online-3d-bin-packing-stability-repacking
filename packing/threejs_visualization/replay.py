"""Export self-contained, movable Three.js replay directories."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Any

from .assets import validated_runtime_assets
from .frame import VisualizationFrame


def normalize_three_replay_path(path: str | Path) -> Path:
    """Return the replay directory, removing a trailing ``.html`` suffix."""
    normalized = Path(path)
    if normalized.suffix.lower() == ".html":
        normalized = normalized.with_suffix("")
    return normalized


def _json_snapshot(payload: Any) -> Any:
    """Validate strict JSON and return a detached JSON-native copy."""
    try:
        encoded = json.dumps(payload, allow_nan=False)
    except TypeError as error:
        raise TypeError(f"replay data must be JSON serializable: {error}") from error
    except ValueError as error:
        if "Out of range float values" in str(error):
            raise ValueError(f"replay numbers must be finite: {error}") from error
        raise TypeError(f"replay data must be JSON serializable: {error}") from error
    return json.loads(encoded)


class ReplayPublishRecoveryError(RuntimeError):
    """A publish failed and automatic rollback left recoverable files."""

    def __init__(
        self,
        original_error: BaseException,
        rollback_errors: list[BaseException],
        recovery_dir: Path,
    ) -> None:
        details = "; ".join(str(error) for error in rollback_errors)
        super().__init__(
            f"replay publish failed: {original_error}; rollback failed: {details}; "
            f"recovery data retained at {recovery_dir}"
        )
        self.original_error = original_error
        self.rollback_errors = tuple(rollback_errors)
        self.recovery_dir = recovery_dir


class _RollbackFailure(Exception):
    def __init__(
        self, original_error: BaseException, rollback_errors: list[BaseException]
    ) -> None:
        self.original_error = original_error
        self.rollback_errors = rollback_errors


class ThreeReplayRecorder:
    """Collect visualization frames and export a local-file replay bundle."""

    def __init__(
        self,
        out_dir: str | Path,
        interval_ms: int = 700,
        *,
        assets_root: str | Path | None = None,
    ) -> None:
        if isinstance(interval_ms, bool) or not isinstance(interval_ms, int):
            raise TypeError("interval_ms must be a positive integer")
        if interval_ms <= 0:
            raise ValueError("interval_ms must be a positive integer")
        self.out_dir = normalize_three_replay_path(out_dir)
        self.interval_ms = interval_ms
        self.assets_root = assets_root
        self.frames: list[dict[str, Any]] = []
        self._save_lock = threading.Lock()

    def capture(self, title: str, frame: VisualizationFrame | dict) -> None:
        """Append a detached, strictly JSON-compatible replay frame."""
        if not isinstance(title, str):
            raise TypeError("title must be a string")
        if not title.strip():
            raise ValueError("title must not be empty")
        if isinstance(frame, VisualizationFrame):
            payload = frame.to_dict()
        elif isinstance(frame, dict):
            payload = frame
        else:
            raise TypeError("frame must be a VisualizationFrame or dict")

        record = _json_snapshot({"title": title, "frame": payload})
        self.frames.append(record)

    def save(self) -> Path:
        """Write owned replay files while preserving unrelated output files.

        The instance lock serializes calls made through this recorder. Because
        the documented layout keeps unrelated files at fixed paths, readers in
        other processes may still observe the brief multi-file publish window.
        """
        with self._save_lock:
            return self._save_locked()

    def _save_locked(self) -> Path:
        assets = validated_runtime_assets(self.assets_root)
        replay = _json_snapshot(
            {"interval_ms": self.interval_ms, "frames": self.frames}
        )
        self.out_dir.parent.mkdir(parents=True, exist_ok=True)
        prefix = f".{self.out_dir.name}.stage-"
        workspace = Path(tempfile.mkdtemp(prefix=prefix, dir=self.out_dir.parent))
        retain_workspace = False
        try:
            staged = workspace / "bundle"
            backup = workspace / "backup"
            self._stage_bundle(staged, assets, replay)
            try:
                self._publish_bundle(staged, backup)
            except _RollbackFailure as failure:
                recovery_dir = self._retain_recovery_workspace(workspace)
                retain_workspace = True
                raise ReplayPublishRecoveryError(
                    failure.original_error,
                    failure.rollback_errors,
                    recovery_dir,
                ) from failure.original_error
            return self.out_dir / "index.html"
        finally:
            if not retain_workspace and os.path.lexists(workspace):
                shutil.rmtree(workspace)

    def _retain_recovery_workspace(self, workspace: Path) -> Path:
        recovery_name = workspace.name.replace(".stage-", ".recovery-", 1)
        recovery_dir = workspace.with_name(recovery_name)
        try:
            os.rename(workspace, recovery_dir)
        except OSError:
            # The original workspace remains persistent and is still named in
            # the raised recovery error if even the same-filesystem rename fails.
            return workspace
        return recovery_dir

    @staticmethod
    def _stage_bundle(staged: Path, assets, replay: dict[str, Any]) -> None:
        """Fully prepare a candidate bundle without touching the destination."""
        static_dir = staged / "static"
        licenses_dir = static_dir / "licenses"
        licenses_dir.mkdir(parents=True)
        shutil.copy2(assets.replay_html, staged / "index.html")
        shutil.copy2(assets.renderer_js, static_dir / "renderer.bundle.js")
        shutil.copy2(assets.style_css, static_dir / "style.css")
        shutil.copy2(
            assets.three_license,
            licenses_dir / "three-LICENSE.txt",
        )
        frames_js = (
            "window.PACKING_THREE_REPLAY = "
            + json.dumps(replay, allow_nan=False)
            + ";\n"
        )
        (staged / "frames.js").write_text(frames_js, encoding="utf-8")

    def _publish_bundle(self, staged: Path, backup: Path) -> None:
        """Publish owned files, restoring the previous bundle on any failure."""
        owned_paths = (
            Path("index.html"),
            Path("frames.js"),
            Path("static/renderer.bundle.js"),
            Path("static/style.css"),
            Path("static/licenses/three-LICENSE.txt"),
        )
        required_dirs = (
            self.out_dir,
            self.out_dir / "static",
            self.out_dir / "static" / "licenses",
        )
        created_dirs: list[Path] = []
        backed_up: set[Path] = set()
        published: set[Path] = set()

        try:
            for directory in required_dirs:
                if os.path.lexists(directory):
                    if directory.is_symlink():
                        raise ValueError(
                            f"replay output directory must not be a symbolic link: "
                            f"{directory}"
                        )
                    if not directory.is_dir():
                        raise NotADirectoryError(
                            f"replay output directory is not a directory: {directory}"
                        )
                else:
                    directory.mkdir()
                    created_dirs.append(directory)

            for relative in owned_paths:
                destination = self.out_dir / relative
                if destination.is_symlink():
                    raise ValueError(
                        f"replay output file must not be a symbolic link: "
                        f"{destination}"
                    )
                if os.path.lexists(destination) and destination.is_dir():
                    raise IsADirectoryError(
                        f"replay output file is a directory: {destination}"
                    )

            for relative in owned_paths:
                source = staged / relative
                destination = self.out_dir / relative
                previous = backup / relative
                if os.path.lexists(destination):
                    previous.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(destination, previous)
                    backed_up.add(relative)
                os.replace(source, destination)
                published.add(relative)
        except BaseException as publish_error:
            rollback_errors: list[BaseException] = []
            for relative in reversed(owned_paths):
                destination = self.out_dir / relative
                previous = backup / relative
                removal_error: BaseException | None = None
                if relative in published and os.path.lexists(destination):
                    try:
                        destination.unlink()
                    except BaseException as error:
                        removal_error = error
                if relative in backed_up and os.path.lexists(previous):
                    try:
                        os.replace(previous, destination)
                    except BaseException as error:
                        if removal_error is not None:
                            rollback_errors.append(removal_error)
                            removal_error = None
                        rollback_errors.append(error)
                    else:
                        removal_error = None
                if removal_error is not None:
                    rollback_errors.append(removal_error)
            for directory in reversed(created_dirs):
                try:
                    directory.rmdir()
                except OSError:
                    pass
            if rollback_errors:
                raise _RollbackFailure(
                    publish_error, rollback_errors
                ) from publish_error
            raise
