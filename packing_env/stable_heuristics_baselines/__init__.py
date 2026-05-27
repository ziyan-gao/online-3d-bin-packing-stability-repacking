from .convex_hull import (
    ConvexHullBaseline,
    ConvexHullOldBaseline,
    ConvexHullPlainBaseline,
)
from .adaptive_tree import AdaptiveTreeBaseline
from .combined_rules import CombinedRulesBaseline
from .utils import build_height_bound_candidates, make_item


BASELINES = {
    "convex_hull": ConvexHullBaseline,
    "convex_hull_old": ConvexHullOldBaseline,
    "convex_hull_plain": ConvexHullPlainBaseline,
    "adaptive_tree": AdaptiveTreeBaseline,
    "combined_rules": CombinedRulesBaseline,
}


__all__ = [
    "AdaptiveTreeBaseline",
    "BASELINES",
    "CombinedRulesBaseline",
    "ConvexHullBaseline",
    "ConvexHullOldBaseline",
    "ConvexHullPlainBaseline",
    "build_height_bound_candidates",
    "make_item",
]
