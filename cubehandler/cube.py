"""
Routines regarding gaussian cube files
"""

import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import ase
import numpy as np
from skimage import transform

ANG_TO_BOHR = 1.8897259886
SUPPORTED_IMAGE_FORMATS = ("png", "jpg", "jpeg", "tif", "tiff", "bmp", "pnm", "ps")
_PIL_IMAGE_FORMATS = {
    "png": "PNG",
    "jpg": "JPEG",
    "jpeg": "JPEG",
    "tif": "TIFF",
    "tiff": "TIFF",
    "bmp": "BMP",
    "pnm": "PPM",
    "ps": "EPS",
}


@dataclass(frozen=True)
class RenderSpec:
    orientation16: tuple[float, ...]
    iso_pairs: tuple[tuple[float, str], ...]
    image_format: str
    output_path: Path
    width: int = 1600
    height: int = 1200
    background: str = "#FFFFFF"
    surface_opacity: float = 0.55
    atom_scale: float = 0.8
    bond_radius: float = 0.1


def remove_trailing_zeros(number):
    # Format the number using fixed-point notation with high precision
    number_str = f"{number:.11f}"

    # Remove trailing zeros and a possible trailing decimal point
    number_str = number_str.rstrip("0").rstrip(".")

    # Return the cleaned-up number as a string to avoid automatic conversion to scientific notation
    return number_str


def remove_useless_zeros(input_string):
    # Regular expression to match floating-point numbers
    float_pattern = re.compile(r"\b\d+\.\d+\b")

    def replace_float(match):
        float_str = match.group(0)
        cleaned_float_str = remove_trailing_zeros(float(float_str))
        return cleaned_float_str

    # Replace all occurrences of floating-point numbers in the input string
    return float_pattern.sub(replace_float, input_string)


class Cube:
    """Gaussian cube format."""

    default_origin = np.array([0.0, 0.0, 0.0])
    default_scaling_factor = 1.0
    default_comment = "Cubehandler"
    default_low_precision_decimals = 3

    def __init__(
        self,
        title=None,
        comment=default_comment,
        ase_atoms=None,
        origin=default_origin,
        scaling_factor=default_scaling_factor,
        low_precision_decimals=default_low_precision_decimals,
        cell=None,
        cell_n=None,
        data=None,
    ):
        """
        cell in [au] and (3x3)
        origin in [au]
        """
        self.title = title
        self.comment = comment
        self.ase_atoms = ase_atoms
        self.origin = origin
        self.scaling_factor = scaling_factor
        self.cell = cell
        self.data = data
        self.low_precision_decimals = low_precision_decimals
        if data is not None:
            self.cell_n = data.shape
        else:
            self.cell_n = cell_n

    @classmethod
    def from_file_handle(cls, filehandle, read_data=True, apply_scaling=True):
        f = filehandle
        c = cls()
        c.title = f.readline().rstrip()
        c.comment = f.readline().rstrip()
        if "Scaling factor:" in c.comment:
            c.scaling_factor = float(c.comment.split()[-1])

        line = f.readline().split()
        natoms = int(line[0])

        section_headers = False
        if natoms < 0:
            # A negative number of atoms usually indicates that there
            # are multiple data sections and each of those have a header
            natoms = -natoms
            section_headers = True

        c.origin = np.array(line[1:], dtype=float)

        c.cell_n = np.empty(3, dtype=int)
        c.cell = np.empty((3, 3))
        for i in range(3):
            n, x, y, z = (float(s) for s in f.readline().split())
            c.cell_n[i] = int(n)
            c.cell[i] = n * np.array([x, y, z])

        numbers = np.empty(natoms, int)
        positions = np.empty((natoms, 3))
        for i in range(natoms):
            line = f.readline().split()
            numbers[i] = int(line[0])
            positions[i] = [float(s) for s in line[2:]]

        positions /= ANG_TO_BOHR  # convert from bohr to ang

        c.ase_atoms = ase.Atoms(
            numbers=numbers, positions=positions, cell=c.cell / ANG_TO_BOHR
        )

        if read_data:
            # Option 1: less memory usage but might be slower
            c.data = np.empty(c.cell_n[0] * c.cell_n[1] * c.cell_n[2], dtype=float)
            cursor = 0
            if section_headers:
                f.readline()

            for line in f:
                ls = line.split()
                c.data[cursor : cursor + len(ls)] = ls
                cursor += len(ls)

            # Option 2: Takes much more memory (but may be faster)
            # data = np.array(f.read().split(), dtype=float)

            c.data = c.data.reshape(c.cell_n)
            if apply_scaling:
                c.data *= c.scaling_factor

        return c

    @classmethod
    def from_file(cls, filepath, read_data=True, apply_scaling=True):
        with open(filepath) as f:
            c = cls.from_file_handle(
                f, read_data=read_data, apply_scaling=apply_scaling
            )
        return c

    @classmethod
    def from_content(cls, content, read_data=True, apply_scaling=True):
        return cls.from_file_handle(
            io.StringIO(content), read_data=read_data, apply_scaling=apply_scaling
        )

    def write_cube_file(self, filename, low_precision=False):
        scaling_factor = 1.0
        data = self.data.copy()

        if low_precision:
            scaling_factor = self.rescale_data(data)
            data = np.round(data, decimals=self.low_precision_decimals)

            # Convert -0 to 0
            data[data == 0] = 0

        f = open(filename, "w")

        # Write the title.
        if self.title is None:
            f.write(filename + "\n")
        else:
            f.write(self.title + "\n")

        # Write the comment.
        if "Scaling factor:" in self.comment:
            self.comment = re.sub(
                r"Scaling factor: \d+\.\d+",
                f"Scaling factor: {scaling_factor}",
                self.comment,
            )
        else:
            self.comment += f" Scaling factor: {scaling_factor}"
        f.write(self.comment + "\n")

        natoms = len(self.ase_atoms)

        # Write the number of atoms and the origin.
        f.write(
            f"{natoms:5d} {self.origin[0]:12.6f} {self.origin[1]:12.6f} {self.origin[2]:12.6f}\n"
        )

        # Write the cell dimensions.
        dv_br = self.cell / data.shape
        for i in range(3):
            f.write(
                f"{data.shape[i]:5d} {dv_br[i][0]:12.6f} {dv_br[i][1]:12.6f} {dv_br[i][2]:12.6f}\n"
            )

        # Write the atom positions.
        pos = self.ase_atoms.positions * ANG_TO_BOHR
        numbers = self.ase_atoms.get_atomic_numbers()
        for i in range(natoms):
            f.write(
                f"{numbers[i]:5d} {0.0:12.6f} {pos[i, 0]:12.6f} {pos[i, 1]:12.6f} {pos[i, 2]:12.6f}\n"
            )

        # Write the data.
        if low_precision:
            string_io = io.StringIO()
            format_string = f"%.{self.low_precision_decimals}f"
            np.savetxt(string_io, data.flatten(), fmt=format_string)
            result_string = remove_useless_zeros(string_io.getvalue())
            f.write(result_string)
        else:
            data.tofile(f, sep="\n", format="%12.6e")

        f.close()

    def reduce_data_density_slicing(self, points_per_angstrom=2):
        """Reduces the data density"""
        # We should have ~ 1 point per Bohr
        slicer = np.round(
            1.0 / (points_per_angstrom * np.linalg.norm(self.dcell, axis=1))
        ).astype(int)
        try:
            self.data = self.data[:: slicer[0], :: slicer[1], :: slicer[2]]
        except ValueError:
            print("Warning: Could not reduce data density")

    def reduce_data_density_skimage(self, resolution_factor=0.4):
        new_shape = tuple(int(dim * resolution_factor) for dim in self.data.shape)
        self.data = transform.resize(self.data, new_shape, anti_aliasing=True)

    def render(self, spec: RenderSpec) -> Path:
        """Render a cube file using a camera orientation from nglview."""
        import pyvista as pv
        from PIL import Image
        from ase import data as ase_data
        from ase.data import colors as ase_colors

        image_format = spec.image_format.lower()
        if image_format not in SUPPORTED_IMAGE_FORMATS:
            raise ValueError(
                f"Unsupported image format '{spec.image_format}'. "
                f"Supported formats: {', '.join(SUPPORTED_IMAGE_FORMATS)}."
            )
        if not spec.iso_pairs:
            raise ValueError("At least one isovalue/color pair must be provided.")

        output_path = Path(spec.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        rotation, translation, scale = self._decompose_ngl_orientation(
            spec.orientation16
        )
        structured_grid = self._build_structured_grid(rotation, translation)

        plotter = pv.Plotter(off_screen=True, window_size=(spec.width, spec.height))
        plotter.set_background(spec.background)

        for isovalue, color in spec.iso_pairs:
            contour = structured_grid.contour(
                isosurfaces=[isovalue], scalars="cube_data"
            )
            if contour.n_points:
                plotter.add_mesh(
                    contour,
                    color=color,
                    opacity=spec.surface_opacity,
                    smooth_shading=True,
                )

        transformed_atom_positions = self._transform_points(
            self.ase_atoms.positions, rotation, translation
        )
        atom_numbers = self.ase_atoms.get_atomic_numbers()
        for number, position in zip(atom_numbers, transformed_atom_positions):
            radius = float(ase_data.covalent_radii[number] * spec.atom_scale)
            sphere = pv.Sphere(
                radius=radius,
                center=tuple(position),
                theta_resolution=24,
                phi_resolution=24,
            )
            atom_color = tuple(float(c) for c in ase_colors.cpk_colors[number])
            plotter.add_mesh(sphere, color=atom_color, smooth_shading=True)

        bond_points, _ = self._compute_bonds(self.ase_atoms)
        if len(bond_points):
            transformed_bond_points = self._transform_points(
                bond_points.reshape(-1, 3), rotation, translation
            ).reshape(bond_points.shape)
            for start, end in transformed_bond_points:
                line = pv.Line(start, end)
                tube = line.tube(radius=spec.bond_radius)
                plotter.add_mesh(tube, color="#A0A0A0", smooth_shading=True)

        plotter.camera.position = (0.0, 0.0, -scale)
        plotter.camera.focal_point = (0.0, 0.0, 0.0)
        plotter.camera.up = (0.0, 1.0, 0.0)
        image_array = plotter.screenshot(return_img=True)
        plotter.close()

        image = Image.fromarray(image_array)
        pil_format = _PIL_IMAGE_FORMATS[image_format]
        if pil_format in {"JPEG", "EPS"} and image.mode in {"RGBA", "LA"}:
            image = image.convert("RGB")
        image.save(output_path, format=pil_format)
        return output_path

    @staticmethod
    def _transform_points(
        points: np.ndarray,
        rotation: np.ndarray,
        translation: np.ndarray,
    ) -> np.ndarray:
        points = np.asarray(points, dtype=float)
        return points @ rotation.T + translation

    @staticmethod
    def _decompose_ngl_orientation(
        orientation16: Sequence[float],
    ) -> tuple[np.ndarray, np.ndarray, float]:
        orientation_values = np.asarray(orientation16, dtype=float)
        if orientation_values.size != 16:
            raise ValueError("Camera orientation must contain exactly 16 numbers.")
        if not np.all(np.isfinite(orientation_values)):
            raise ValueError("Camera orientation must contain only finite numbers.")

        matrix = orientation_values.reshape((4, 4), order="F")
        affine = matrix[:3, :3]
        translation = matrix[:3, 3]

        column_norms = np.linalg.norm(affine, axis=0)
        scale = float(np.mean(column_norms))
        if not np.isfinite(scale) or scale <= 1e-12:
            raise ValueError("Could not derive camera scale from orientation matrix.")

        rotation = affine / scale
        if np.linalg.matrix_rank(rotation) < 3:
            raise ValueError("Camera orientation matrix is singular.")

        u_mat, _, vh_mat = np.linalg.svd(rotation)
        rotation = u_mat @ vh_mat
        if np.linalg.det(rotation) < 0:
            u_mat[:, -1] *= -1
            rotation = u_mat @ vh_mat

        if not np.all(np.isfinite(rotation)):
            raise ValueError("Camera orientation produced non-finite rotation values.")
        return rotation, translation, scale

    def _build_structured_grid(
        self,
        rotation: np.ndarray,
        translation: np.ndarray,
    ):
        import pyvista as pv

        nx, ny, nz = (int(v) for v in self.cell_n)
        x_idx, y_idx, z_idx = np.meshgrid(
            np.arange(nx, dtype=float),
            np.arange(ny, dtype=float),
            np.arange(nz, dtype=float),
            indexing="ij",
        )
        origin_ang = self.origin / ANG_TO_BOHR
        voxel_vectors_ang = self.dcell_ang

        coords = (
            origin_ang
            + x_idx[..., None] * voxel_vectors_ang[0]
            + y_idx[..., None] * voxel_vectors_ang[1]
            + z_idx[..., None] * voxel_vectors_ang[2]
        )
        transformed_coords = self._transform_points(
            coords.reshape(-1, 3), rotation, translation
        ).reshape(coords.shape)

        grid = pv.StructuredGrid(
            transformed_coords[..., 0],
            transformed_coords[..., 1],
            transformed_coords[..., 2],
        )
        grid["cube_data"] = np.asarray(self.data, dtype=float).ravel(order="F")
        return grid

    def _compute_bonds(self, structure):
        """Create an list of bonds for the structure."""

        import ase.neighborlist
        from ase.data import colors

        if len(structure) <= 1:
            return np.empty((0, 2, 3), dtype=float), np.empty((0, 3), dtype=float)

        # The value 1.09 is chosen based on our experience. It is a good compromise between showing too many bonds
        # and not showing bonds that should be there.
        cutoff = ase.neighborlist.natural_cutoffs(structure, mult=1.09)

        ii, bond_vectors = ase.neighborlist.neighbor_list(
            "iD", structure, cutoff, self_interaction=False
        )
        # bond start position
        v1 = structure.positions[ii]
        # middle position
        v2 = v1 + bond_vectors * 0.5
        points = np.stack((v1, v2), axis=1)

        # Choose the correct way for computing the cylinder.
        numbers = structure.get_atomic_numbers()

        return points, colors.cpk_colors[numbers[ii]]

    def rescale_data(self, data):
        """Rescales the data to be between -1 and 1."""
        scaling_factor = max(abs(data.min()), abs(data.max()))
        data /= scaling_factor
        return scaling_factor

    def swapaxes(self, ax1, ax2):
        p = self.ase_atoms.positions
        p[:, ax1], p[:, ax2] = p[:, ax2], p[:, ax1].copy()

        self.origin[ax1], self.origin[ax2] = (self.origin[ax2], self.origin[ax1].copy())

        self.cell[:, ax1], self.cell[:, ax2] = (
            self.cell[:, ax2],
            self.cell[:, ax1].copy(),
        )
        self.cell[ax1, :], self.cell[ax2, :] = (
            self.cell[ax2, :],
            self.cell[ax1, :].copy(),
        )

        self.data = np.swapaxes(self.data, ax1, ax2)

        self.cell_n = self.data.shape

    def get_plane_above_topmost_atom(self, height, axis=2):
        """
        Returns the 2d plane above topmost atom in direction (default: z)
        height in [angstrom]
        """
        topmost_atom_z = np.max(self.ase_atoms.positions[:, axis])  # Angstrom
        plane_z = (height + topmost_atom_z) * ANG_TO_BOHR - self.origin[axis]

        plane_index = int(
            np.round(
                plane_z / self.cell[axis, axis] * np.shape(self.data)[axis] - 0.499
            )
        )

        if axis == 0:
            return self.data[plane_index, :, :]
        if axis == 1:
            return self.data[:, plane_index, :]
        return self.data[:, :, plane_index]

    def get_x_index(self, x_ang):
        # returns the index value for a given x coordinate in angstrom
        return int(
            np.round(
                (x_ang * ANG_TO_BOHR - self.origin[0])
                / self.cell[0, 0]
                * np.shape(self.data)[0]
            )
        )

    def get_y_index(self, y_ang):
        # returns the index value for a given y coordinate in angstrom
        return int(
            np.round(
                (y_ang * ANG_TO_BOHR - self.origin[1])
                / self.cell[1, 1]
                * np.shape(self.data)[1]
            )
        )

    def get_z_index(self, z_ang):
        # returns the index value for a given z coordinate in angstrom
        return int(
            np.round(
                (z_ang * ANG_TO_BOHR - self.origin[2])
                / self.cell[2, 2]
                * np.shape(self.data)[2]
            )
        )

    @property
    def integral(self):
        """Computes the integral of the cube data."""
        return np.sum(self.data) * self.dv_au

    @property
    def dv(self):
        """in [ang]"""
        return self.dv_ang

    @property
    def dv_ang(self):
        """in [ang]"""
        return self.ase_atoms.get_volume() / self.data.size

    @property
    def dv_au(self):
        """in [au]"""
        return ANG_TO_BOHR**3 * self.dv_ang

    @property
    def dcell(self):
        """in [ang]"""
        return self.dcell_ang

    @property
    def dcell_ang(self):
        """in [ang]"""
        return self.dcell_au / ANG_TO_BOHR

    @property
    def dcell_au(self):
        """in [au]"""
        return self.cell / self.cell_n

    @property
    def x_arr_au(self):
        """in [au]"""
        return np.arange(
            self.origin[0],
            self.origin[0] + (self.cell_n[0] - 0.5) * self.dcell_au[0, 0],
            self.dcell_au[0, 0],
        )

    @property
    def y_arr_au(self):
        """in [au]"""
        return np.arange(
            self.origin[1],
            self.origin[1] + (self.cell_n[1] - 0.5) * self.dcell_au[1, 1],
            self.dcell_au[1, 1],
        )

    @property
    def z_arr_au(self):
        """in [au]"""
        return np.arange(
            self.origin[2],
            self.origin[2] + (self.cell_n[2] - 0.5) * self.dcell_au[2, 2],
            self.dcell_au[2, 2],
        )

    @property
    def x_arr_ang(self):
        """in [ang]"""
        return self.x_arr_au / ANG_TO_BOHR

    @property
    def y_arr_ang(self):
        """in [ang]"""
        return self.y_arr_au / ANG_TO_BOHR

    @property
    def z_arr_ang(self):
        """in [ang]"""
        return self.z_arr_au / ANG_TO_BOHR
