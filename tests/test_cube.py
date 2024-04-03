import os
import numpy as np
from cubehandler import Cube

this_dir = os.path.dirname(os.path.abspath(__file__))


def test_empty_cube():
    cube = Cube()
    assert cube is not None


def test_read_cube():
    cube = Cube.from_file(this_dir + "/CH4_HOMO.cube")
    assert np.allclose(cube.origin, np.array([1.0, 0, 0]), atol=0.001)
    assert np.allclose(
        cube.cell,
        np.array([[22.1019, 0, 0], [0, 22.0465, 0], [0, 0, 22.1979]]),
        atol=0.001,
    )


def test_get_xyz_index():
    cube = Cube.from_file(this_dir + "/CH4_HOMO.cube")
    assert cube.get_x_index(1) == 2
    assert cube.get_y_index(1) == 5
    assert cube.get_z_index(1) == 5
