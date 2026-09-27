# LP analysis and visualization migration

User-approved scope: migrate the LP solver, visualization, runnable example, and a brief README explanation.

Reuse the target repository's Item and Heu_Stable support cache. Add lbcp_polygon_map() to expose cached world-coordinate polygons. Migrate the current source LP modules and Three.js backend/assets; add missing visualization configuration fields without replacing target defaults or its existing renderer. Provide packing.lp_analysis for analyzing a live PackingEnv and building a force overlay, and packing.lp_demo for an offline replay or localhost live viewer.

LP inputs use mm, kg/mm³, m/s²; forces and payload bounds are in N. Solve vertical force and XY moment balance with nonnegative contact forces and LBCP-clipped interfaces. Default COM is geometric center. Bounds are independent extrema, not one simultaneous force distribution. Surface geometry and LP failure status remain visible; no force arrows are fabricated for infeasible states. All item-to-item and ground interfaces are visualized.

Keep existing policy, candidate filtering, repacking and default visualization behavior. Users call the new analysis API on their existing environment or use the standalone demo. Recompute after packing/unpacking; reject missing cached LBCP data rather than silently assuming full support.

Validation: existing target suite (baseline: 88 passed), migrated LP and visualization tests, adapter integration tests for cache lifecycle/units/force balance/status, offline export and CLI smoke runs, JavaScript tests and bundle build. No runtime dependency on neuromeka-base; include frontend runtime assets and third-party license.
