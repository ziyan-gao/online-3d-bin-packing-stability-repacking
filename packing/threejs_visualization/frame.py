"""Renderer-neutral records for three-scene packing visualizations.

Records are assignment-frozen, while nested JSON containers are builder-owned.
``to_dict()`` returns an independent recursive copy for consumers.
"""

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class SceneObject:
    id: str
    kind: str
    geometry: dict[str, Any]
    style: dict[str, Any]
    tooltip: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SceneFrame:
    bounds: list[float]
    objects: list[SceneObject]
    camera: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VisualizationFrame:
    title: str
    scenes: dict[str, SceneFrame]
    legend: list[dict[str, Any]]
    schema_version: int = 1
    metrics: dict[str, float | int | str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
