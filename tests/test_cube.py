from pathlib import Path

import numpy as np

from cubehandler import Cube

this_file_path = Path(__file__)
this_dir = this_file_path.parent.absolute()


def test_empty_cube():
    cube = Cube()
    assert cube is not None


def test_read_cube():
    cube = Cube.from_file(this_dir / "CH4_HOMO.cube")
    assert np.allclose(cube.origin, np.array([1.0, 0, 0]), atol=0.001)
    assert np.allclose(
        cube.cell,
        np.array([[15.1178, 0, 0], [0, 15.1178, 0], [0, 0, 15.1178]]),
        atol=0.001,
    )


def test_get_xyz_index():
    cube = Cube.from_file(this_dir / "CH4_HOMO.cube")
    assert cube.get_x_index(1) == 2
    assert cube.get_y_index(1) == 4
    assert cube.get_z_index(1) == 4
