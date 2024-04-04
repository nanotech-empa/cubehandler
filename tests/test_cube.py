import os
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


def test_dv():
    cube = Cube.from_file(this_dir / "CH4_HOMO.cube")
    assert np.allclose(
        cube.dv_ang, np.array([[0.25, 0, 0], [0, 0.25, 0], [0, 0, 0.25]]), atol=0.001
    )


def test_x_arr_ang():
    cube = Cube.from_file(this_dir / "CH4_HOMO.cube")
    assert np.allclose(
        cube.x_arr_ang,
        np.array(
            [
                0.52917725,
                0.77917752,
                1.02917778,
                1.27917805,
                1.52917831,
                1.77917858,
                2.02917885,
                2.27917911,
                2.529179,
                2.77917927,
                3.02917954,
                3.27917981,
                3.52918008,
                3.77918035,
                4.02918062,
                4.27918089,
                4.52918116,
                4.77918143,
                5.0291817,
                5.27918197,
                5.52918224,
                5.77918251,
                6.02918278,
                6.27918305,
                6.52918332,
                6.77918359,
                7.02918386,
                7.27918413,
                7.5291844,
                7.77918467,
                8.02918494,
                8.27918521,
            ]
        ),
        atol=0.001,
    )
