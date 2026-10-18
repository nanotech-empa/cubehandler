"""Sampling a rendered plane with the shared EMPA viewer's conventions."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SlicePlane:
    """Plane coordinates in angstrom, with a frozen colour range."""

    center: tuple[float, float, float]
    normal: tuple[float, float, float]
    offset: float
    width: float
    height: float
    resolution: int
    vmin: float
    vmax: float
    periodic: bool = False

    def __post_init__(self):
        if any(
            np.shape(v) != (3,) or not np.isfinite(v).all()
            for v in (self.center, self.normal)
        ):
            raise ValueError(
                "Plane center and normal must contain three finite numbers"
            )
        if np.linalg.norm(self.normal) < 1e-12:
            raise ValueError("Plane normal must be nonzero")
        if not np.isfinite(
            [self.offset, self.width, self.height, self.vmin, self.vmax]
        ).all():
            raise ValueError("Plane dimensions and colour limits must be finite")
        if self.width <= 0 or self.height <= 0 or self.vmin > self.vmax:
            raise ValueError("Plane dimensions must be positive and vmin <= vmax")
        if type(self.resolution) is not int or not 16 <= self.resolution <= 1024:
            raise ValueError("Plane resolution must be an integer between 16 and 1024")
        if type(self.periodic) is not bool:
            raise ValueError("Plane periodic flag must be boolean")

    def sample(self, cube):
        """Use the viewer's linear interpolation, origin and boundary convention."""
        from scipy.ndimage import map_coordinates
        from .cube import ANG_TO_BOHR

        normal = np.array(self.normal, dtype=float, copy=True)
        normal /= np.linalg.norm(normal)
        trial = (
            np.array([1.0, 0.0, 0.0])
            if abs(normal[0]) < 0.9
            else np.array([0.0, 1.0, 0.0])
        )
        u = np.cross(normal, trial)
        u /= np.linalg.norm(u)
        v = np.cross(normal, u)
        us, vs = np.meshgrid(
            np.linspace(-self.width / 2, self.width / 2, self.resolution),
            np.linspace(-self.height / 2, self.height / 2, self.resolution),
        )
        grid = (
            np.asarray(self.center)
            + self.offset * normal
            + us[..., None] * u
            + vs[..., None] * v
        )
        positions = grid.reshape(-1, 3) - cube.origin / ANG_TO_BOHR
        fractional = positions @ np.linalg.inv(cube.ase_atoms.cell.array)
        indexes = (fractional * np.asarray(cube.data.shape)).reshape(grid.shape)
        values = map_coordinates(
            cube.data,
            np.moveaxis(indexes, -1, 0),
            order=1,
            mode="wrap" if self.periodic else "constant",
            cval=np.nan,
            prefilter=False,
        )
        return grid, values

    def mesh(self, cube, rotation, translation):
        """Colour the plane and omit cells outside an isolated cube's grid."""
        import matplotlib as mpl
        from matplotlib.colors import Normalize
        import pyvista as pv

        points, values = self.sample(cube)
        transformed = cube._transform_points(
            points.reshape(-1, 3), rotation, translation
        ).reshape(points.shape)
        mesh = pv.StructuredGrid(
            transformed[..., 0], transformed[..., 1], transformed[..., 2]
        )
        rgb = mpl.colormaps["coolwarm"](Normalize(self.vmin, self.vmax)(values))[
            ..., :3
        ]
        mesh["slice_rgb"] = np.round(rgb.reshape(-1, 3, order="F") * 255).astype(
            np.uint8
        )
        finite = np.isfinite(values)
        valid = finite[:-1, :-1] & finite[1:, :-1] & finite[:-1, 1:] & finite[1:, 1:]
        return mesh.extract_cells(
            np.flatnonzero(valid.ravel(order="F"))
        ).extract_surface()
