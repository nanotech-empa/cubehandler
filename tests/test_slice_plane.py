"""Plane geometry, isolated boundaries, and visible-plane render regression."""

from dataclasses import replace

import numpy as np
import pytest
from ase import Atoms

from cubehandler import Cube, RenderSpec
from cubehandler.cube import ANG_TO_BOHR
from cubehandler.slice_plane import SlicePlane


def example():
    shape = (8, 10, 12)
    cell = np.array([[4.0, 0.0, 0.0], [1.0, 5.0, 0.0], [0.3, 0.5, 6.0]])
    origin = np.array([-2.0, -3.0, -4.0])
    ijk = np.indices(shape).transpose(1, 2, 3, 0)
    xyz = origin + ijk @ (cell / np.array(shape)[:, None])
    data = xyz[..., 0] + 2 * xyz[..., 1] + 3 * xyz[..., 2]
    center = origin + cell.sum(axis=0) / 2
    cube = Cube(
        ase_atoms=Atoms("H", positions=[center], cell=cell),
        origin=origin * ANG_TO_BOHR,
        cell=cell * ANG_TO_BOHR,
        data=data,
    )
    plane = SlicePlane(
        center=center,
        normal=(0.2, 0.3, 1.0),
        offset=0.1,
        width=1.0,
        height=1.0,
        resolution=32,
        vmin=-10.0,
        vmax=10.0,
    )
    return cube, plane


def test_plane_sampling_preserves_origin_skew_and_boundaries():
    cube, plane = example()
    grid, values = plane.sample(cube)
    np.testing.assert_allclose(
        values, grid[..., 0] + 2 * grid[..., 1] + 3 * grid[..., 2], atol=1e-12
    )
    assert np.isnan(replace(plane, center=(100.0, 100.0, 100.0)).sample(cube)[1]).all()
    assert np.isfinite(
        replace(plane, center=(100.0, 100.0, 100.0), periodic=True).sample(cube)[1]
    ).all()
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    shift = np.array([1.0, 2.0, 3.0])
    mesh = plane.mesh(cube, rotation, shift)
    expected = grid.reshape(-1, 3) @ rotation.T + shift
    np.testing.assert_allclose(mesh.points.min(axis=0), expected.min(axis=0))
    np.testing.assert_allclose(mesh.points.max(axis=0), expected.max(axis=0))
    assert mesh.n_cells == (plane.resolution - 1) ** 2
    assert mesh["slice_rgb"].shape == (plane.resolution**2, 3)
    assert (
        replace(plane, center=(100.0, 100.0, 100.0)).mesh(cube, rotation, shift).n_cells
        == 0
    )


def test_plane_is_included_only_when_requested(monkeypatch, tmp_path):
    import pyvista as pv

    cube, plane = example()
    meshes = []

    class Plotter:
        def __init__(self, **kwargs):
            self.camera = type("Camera", (), {})()

        def set_background(self, *args):
            pass

        def add_mesh(self, mesh, **kwargs):
            meshes.append((mesh, kwargs))

        def screenshot(self, **kwargs):
            return np.zeros((32, 32, 3), dtype=np.uint8)

        def close(self):
            pass

    monkeypatch.setattr(pv, "Plotter", Plotter)
    spec = RenderSpec(
        orientation16=tuple((np.eye(4) * 20).ravel()),
        iso_pairs=((0.03, "#FFFFFF"),),
        image_format="png",
        output_path=tmp_path / "image.png",
    )
    cube.render(spec)
    assert not any(kwargs.get("scalars") == "slice_rgb" for _, kwargs in meshes)
    meshes.clear()
    cube.render(replace(spec, slice_plane=plane))
    planes = [
        (mesh, kwargs)
        for mesh, kwargs in meshes
        if kwargs.get("scalars") == "slice_rgb"
    ]
    assert len(planes) == 1 and planes[0][0].n_cells > 0
    assert planes[0][1]["rgb"] and planes[0][1]["opacity"] == 1.0


@pytest.mark.parametrize(
    "kwargs", [dict(normal=(0, 0, 0)), dict(width=0), dict(resolution=0), dict(vmin=20)]
)
def test_invalid_plane_is_reported(kwargs):
    _, plane = example()
    with pytest.raises(ValueError):
        replace(plane, **kwargs)
