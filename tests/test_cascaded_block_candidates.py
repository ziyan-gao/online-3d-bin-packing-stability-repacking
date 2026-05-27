import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from packing_env.data_type import OrientedBlock, Orthogonal3D, Point3D, SimpleBlock


def test_oriented_block_preserves_normal_dimensions_and_metadata():
    block = SimpleBlock(
        box=Orthogonal3D(100, 200, 100),
        stack_dims=(1, 1, 1),
        buffer_space=10,
    )

    candidate = OrientedBlock.from_block(block, source_index=2, rotate_xy=False)

    assert candidate.Dim.raw().tolist() == [100, 200, 100]
    assert candidate.Virtual_Dim.raw().tolist() == [110, 210, 100]
    assert candidate.consumed_count == 1
    assert candidate.source_index == 2
    assert candidate.rotate_xy is False
    assert candidate.feature_row(container_size=(600, 600, 600)).shape == (8,)


def test_oriented_block_to_item_applies_transpose_once():
    block = SimpleBlock(
        box=Orthogonal3D(100, 200, 100),
        stack_dims=(1, 1, 1),
        buffer_space=10,
    )

    placed = OrientedBlock.from_block(
        block,
        source_index=0,
        rotate_xy=True,
    ).to_item(Point3D(10, 20, 3))

    np.testing.assert_array_equal(placed.Dim.raw(), np.array([200, 100, 100]))
    np.testing.assert_array_equal(placed.Virtual_Dim.raw(), np.array([210, 110, 100]))
    assert placed.rot is True
