"""NGL camera coordinates/projection without an OpenGL context or native job.

Reference: NGL ViewerControls.getPositionOnCanvas and getOrientation translate
before rotating; the default shared viewer uses a 40-degree perspective camera.
https://github.com/nglviewer/ngl/blob/master/src/controls/viewer-controls.ts
"""

import numpy as np
import pytest
from ase import Atoms

from cubehandler import Cube, RenderSpec
from cubehandler.cube import ANG_TO_BOHR
from cubehandler.slice_plane import SlicePlane

ROTATION = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


def orientation(rotation, center, distance=18.0):
    matrix = np.eye(4)
    matrix[:3, :3] = rotation * distance
    matrix[:3, 3] = -np.asarray(center)
    return matrix.ravel(order="F")


def example():
    shape = (4, 5, 6)
    cell = np.array([[2.0, 0.0, 0.0], [0.3, 2.0, 0.0], [0.2, 0.1, 2.0]])
    origin = np.array([5.0, 6.0, 7.0])
    center = origin + cell.sum(axis=0) / 2
    cube = Cube(
        ase_atoms=Atoms("HH", positions=[center, center + [0.4, 0.0, 0.0]], cell=cell),
        origin=origin * ANG_TO_BOHR,
        cell=cell * ANG_TO_BOHR,
        data=np.arange(np.prod(shape), dtype=float).reshape(shape) / 100,
    )
    return cube, center


def test_ngl_center_and_basis_are_rotated_after_centering():
    center = np.array([5.7, 5.3, 6.45])
    points = np.array([center, center + [1.0, 0.0, 0.0], center + [0.0, 1.0, 0.0]])
    rotation, shift, distance = Cube._decompose_ngl_orientation(
        orientation(ROTATION, center)
    )
    actual = Cube._transform_points(points, rotation, shift)
    np.testing.assert_allclose(
        actual, [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]], atol=1e-12
    )
    assert distance == pytest.approx(18.0)


def test_grid_atoms_and_visible_plane_share_the_ngl_transform_without_mutation():
    cube, center = example()
    before = [
        cube.data.copy(),
        cube.cell.copy(),
        cube.origin.copy(),
        cube.ase_atoms.positions.copy(),
    ]
    rotation, shift, _ = Cube._decompose_ngl_orientation(orientation(ROTATION, center))
    grid = cube._build_structured_grid(rotation, shift)
    np.testing.assert_allclose(
        grid.points[0], ROTATION @ (cube.origin / ANG_TO_BOHR - center), atol=1e-12
    )
    np.testing.assert_array_equal(grid["cube_data"], cube.data.ravel(order="F"))
    atoms = Cube._transform_points(cube.ase_atoms.positions, rotation, shift)
    np.testing.assert_allclose(atoms, [[0.0, 0.0, 0.0], [0.0, 0.4, 0.0]], atol=1e-12)
    plane = SlicePlane(
        center=center,
        normal=[0.2, 0.3, 1.0],
        offset=0.1,
        width=0.4,
        height=0.4,
        resolution=16,
        vmin=0.0,
        vmax=2.0,
        periodic=True,
    )
    source_points, _ = plane.sample(cube)
    mesh = plane.mesh(cube, rotation, shift)
    expected = (source_points.reshape(-1, 3) - center) @ ROTATION.T
    # VTK surface extraction may reorder vertices without changing geometry.
    from scipy.spatial import cKDTree

    distances, indices = cKDTree(expected).query(mesh.points)
    assert distances.max() < 1e-12
    assert len(set(indices)) == len(expected)
    for got, original in zip(
        [cube.data, cube.cell, cube.origin, cube.ase_atoms.positions], before
    ):
        np.testing.assert_array_equal(got, original)


@pytest.mark.parametrize(
    "rotation,center", [(np.eye(3), [5.0, 6.0, 7.0]), (ROTATION, [0.0, 0.0, 0.0])]
)
def test_identity_rotation_or_zero_center_preserves_existing_coordinates(
    rotation, center
):
    points = np.array([[1.0, 2.0, 3.0], [-4.0, 5.0, 6.0]])
    got_rotation, shift, _ = Cube._decompose_ngl_orientation(
        orientation(rotation, center)
    )
    actual = Cube._transform_points(points, got_rotation, shift)
    np.testing.assert_allclose(actual, (points - center) @ rotation.T, atol=1e-12)


@pytest.mark.parametrize("width,height", [(320, 320), (640, 320), (320, 640)])
def test_renderer_projection_matches_default_ngl_camera_without_rasterization(
    monkeypatch, tmp_path, width, height
):
    import pyvista as pv

    cube, center = example()
    camera = pv.Camera()

    class Plotter:
        def __init__(self, **kwargs):
            assert kwargs["window_size"] == (width, height)
            self.camera = camera

        def set_background(self, *args):
            pass

        def add_mesh(self, *args, **kwargs):
            pass

        def screenshot(self, **kwargs):
            return np.zeros((height, width, 3), dtype=np.uint8)

        def close(self):
            pass

    monkeypatch.setattr(pv, "Plotter", Plotter)
    request = orientation(ROTATION, center, 18.0)
    cube.render(
        RenderSpec(
            orientation16=tuple(request),
            iso_pairs=((0.5, "#336699"),),
            image_format="png",
            output_path=tmp_path / "mock-only.png",
            width=width,
            height=height,
        )
    )
    assert camera.view_angle == pytest.approx(40.0)
    rotation, shift, distance = Cube._decompose_ngl_orientation(request)
    points = np.array([center, center + [1.0, 0.0, 0.0], center + [0.0, 1.0, 0.5]])
    scene = Cube._transform_points(points, rotation, shift)
    aspect = width / height
    matrix = camera.GetCompositeProjectionTransformMatrix(aspect, -1.0, 1.0)
    projection = np.array(
        [[matrix.GetElement(i, j) for j in range(4)] for i in range(4)]
    )
    projected = np.c_[scene, np.ones(len(scene))] @ projection.T
    ndc = projected[:, :2] / projected[:, 3, None]
    ngl_scene = (points - center) @ ROTATION.T
    denominator = (distance + ngl_scene[:, 2]) * np.tan(np.deg2rad(40.0) / 2)
    expected = np.c_[
        -ngl_scene[:, 0] / (denominator * aspect), ngl_scene[:, 1] / denominator
    ]
    np.testing.assert_allclose(ndc, expected, atol=1e-12)
