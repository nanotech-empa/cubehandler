"""
Routines regarding gaussian cube files
"""

import io
import re

import ase
import numpy as np
from skimage import transform

ANG_TO_BOHR = 1.8897259886


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
