"""Synthetic oblique-grid serialization and sample-position regressions."""

import numpy as np
import pytest
from ase import Atoms
from cubehandler import Cube
from cubehandler.cube import ANG_TO_BOHR


def make_cube(cls=Cube, skew=True, shape=(30, 30, 28)):
    cell = (
        np.array([[2.0000041, 0, 0], [-1.000002, 1.7320446, 0], [0, 0, 2.0]])
        if skew
        else np.diag([2.0, 2.0, 2.0])
    )
    atoms = Atoms("BN", positions=[[0.2, 0.3, 0.6], [0.5, 0.7, 0.6]], cell=cell)
    i, j, k = np.indices(shape)
    return cls(
        title="synthetic source",
        comment="density",
        ase_atoms=atoms,
        origin=np.array([0.2, 0.3, 0.4]),
        cell=cell * ANG_TO_BOHR,
        data=(i + 2 * j - 3 * k + 0.5) * 0.0001,
    )


@pytest.mark.parametrize(
    "cell",
    [
        [[2, 0, 0], [-1, 1.732, 0], [0, 0, 3]],
        [[2, 0.2, -0.1], [0.4, 3, 0.3], [-0.2, 0.1, 4]],
        [[2, 0, 0], [0, 3, 0], [0, 0, 4]],
    ],
)
def test_row_vectors_roundtrip_and_voxel_metric(tmp_path, cell):
    c = make_cube(shape=(7, 9, 11))
    c.cell = np.array(cell) * ANG_TO_BOHR
    c.ase_atoms.set_cell(cell)
    np.testing.assert_allclose(c.dcell_au, c.cell / np.array(c.data.shape)[:, None])
    path = tmp_path / "roundtrip.cube"
    c.write_cube_file(path, low_precision=False)
    got = Cube.from_file(path)
    np.testing.assert_allclose(got.cell, c.cell, atol=6e-6, rtol=0)
    np.testing.assert_allclose(got.data, c.data, atol=1e-9, rtol=0)
    np.testing.assert_allclose(
        got.ase_atoms.positions, c.ase_atoms.positions, atol=5e-7, rtol=0
    )
    np.testing.assert_allclose(got.origin, c.origin, atol=1e-6, rtol=0)


@pytest.mark.parametrize("skew", [True, False])
def test_aligned_slicing_preserves_original_cartesian_samples_and_cell(skew):
    c = make_cube(skew=skew)
    data = c.data.copy()
    cell = c.cell.copy()
    origin = c.origin.copy()
    n = np.array(data.shape)
    c.reduce_data_density_slicing()
    new = np.array(c.data.shape)
    steps = n // new
    assert np.all(n % new == 0)
    np.testing.assert_array_equal(c.cell_n, new)
    np.testing.assert_array_equal(c.cell, cell)
    np.testing.assert_array_equal(c.origin, origin)
    np.testing.assert_array_equal(c.data, data[:: steps[0], :: steps[1], :: steps[2]])
    new_i = np.array([new[0] - 1, new[1] - 1, new[2] - 1])
    np.testing.assert_allclose(
        origin + new_i @ c.dcell_au,
        origin + (new_i * steps) @ (cell / n[:, None]),
        atol=1e-13,
    )


def test_skimage_updates_grid_count():
    c = make_cube()
    cell = c.cell.copy()
    c.reduce_data_density_skimage()
    np.testing.assert_array_equal(c.cell_n, c.data.shape)
    np.testing.assert_allclose(c.dcell_au, cell / np.array(c.data.shape)[:, None])
