"""Locations and validation for committed Three.js runtime assets."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RuntimeAssets:
    static_dir: Path
    live_html: Path
    replay_html: Path
    style_css: Path
    renderer_js: Path
    three_license: Path


def validated_runtime_assets(root: str | Path | None = None) -> RuntimeAssets:
    """Return package-local runtime files after checking every required asset."""
    static_dir = (
        Path(root)
        if root is not None
        else Path(__file__).resolve().parent / "frontend" / "static"
    )
    assets = RuntimeAssets(
        static_dir=static_dir,
        live_html=static_dir / "live.html",
        replay_html=static_dir / "replay.html",
        style_css=static_dir / "style.css",
        renderer_js=static_dir / "renderer.bundle.js",
        three_license=static_dir / "licenses" / "three-LICENSE.txt",
    )
    for path in (
        assets.live_html,
        assets.replay_html,
        assets.style_css,
        assets.renderer_js,
        assets.three_license,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"required Three.js runtime asset is missing: {path}")
    return assets
