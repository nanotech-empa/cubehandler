"""
Routines regarding gaussian cube files
"""

import io
import re

import ase
import numpy as np

ANG_TO_BOHR = 1.8897259886


def remove_useless_zeros(s):
    # Pattern to identify unnecessary zeros in a number
    # This pattern looks for numbers that have a decimal point followed by any number of zeros, optionally ending with more digits
    # It captures the number part before the decimal point and the non-zero digits after the decimal point (if any)
    pattern = r"(\d+)\.0+([1-9]*)"

    # Replacement function
    # If there are non-zero digits after the decimal point, keep one trailing zero (for numbers like 1.0, 2.0, etc.)
    # Otherwise, remove the decimal part entirely
    def replacer(match):
        if match.group(2):  # If there are digits after the zeros
            return match.group(1) + "." + match.group(2)  # Keep the non-zero digits
        else:
            return match.group(1)  # Keep only the integer part

    # Use re.sub() to replace the matches of the pattern in the input string with the output of the replacer function

    return re.sub(pattern, replacer, s)


class Cube:
    """
    Gaussian cube format
    """

    default_origin = np.array([0.0, 0.0, 0.0])

    def __init__(
        self,
        title=None,
        comment=None,
        ase_atoms=None,
        origin=default_origin,
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
        self.cell = cell
        self.data = data
        self.scaling_factor = 1.0
        if data is not None:
            self.cell_n = data.shape
        else:
            self.cell_n = cell_n

    @classmethod
    def from_file_handle(cls, filehandle, read_data=True):
        f = filehandle
        c = cls()
        c.title = f.readline().rstrip()
        c.comment = f.readline().rstrip()

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

        return c

    @classmethod
    def from_file(cls, filepath, read_data=True):
        with open(filepath) as f:
            c = cls.from_file_handle(f, read_data=read_data)
        return c

    def write_cube_file(self, filename, low_precision=False):

        natoms = len(self.ase_atoms)

        f = open(filename, "w")

        if self.title is None:
            f.write(filename + "\n")
        else:
            f.write(self.title + "\n")

        if self.comment is None:
            f.write(f"Scaling factor: {self.scaling_factor}\n")
        else:
            f.write(self.comment + f" Scaling factor: {self.scaling_factor}" + "\n")

        dv_br = self.cell / self.data.shape

        f.write(
            "%5d %12.6f %12.6f %12.6f\n"
            % (natoms, self.origin[0], self.origin[1], self.origin[2])
        )

        for i in range(3):
            f.write(
                "%5d %12.6f %12.6f %12.6f\n"
                % (self.data.shape[i], dv_br[i][0], dv_br[i][1], dv_br[i][2])
            )

        if natoms > 0:

            positions = self.ase_atoms.positions * ANG_TO_BOHR
            numbers = self.ase_atoms.get_atomic_numbers()
            for i in range(natoms):
                at_x, at_y, at_z = positions[i]
                f.write(
                    "%5d %12.6f %12.6f %12.6f %12.6f\n"
                    % (numbers[i], 0.0, at_x, at_y, at_z)
                )

        if low_precision:
            string_io = io.StringIO()
            np.savetxt(string_io, self.data.flatten(), fmt="%.3f")
            result_string = remove_useless_zeros(string_io.getvalue())
            f.write(result_string)
        else:
            self.data.tofile(f, sep="\n", format="%12.6e")

        f.close()

    def reduce_data_density(self, points_per_angstrom=2):
        """Reduces the data density"""
        # We should have ~ 1 point per Bohr
        slicer = np.round(
            self.data.shape
            / np.linalg.norm(self.ase_atoms.cell, axis=1)
            / points_per_angstrom
        ).astype(int)
        try:
            self.data = self.data[:: slicer[0], :: slicer[1], :: slicer[2]]
        except ValueError:
            print("Warning: Could not reduce data density")

    def rescale_data(self):
        """Rescales the data to be between -1 and 1"""
        self.scaling_factor = max(abs(self.data.min()), abs(self.data.max()))
        self.data /= self.scaling_factor
        self.data = np.round(self.data, decimals=3)

        # Convert -0 to 0
        self.data[self.data == 0] = 0

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
    def dv(self):
        """in [ang]"""
        return self.cell / self.cell_n / ANG_TO_BOHR

    @property
    def dv_ang(self):
        """in [ang]"""
        return self.cell / self.cell_n / ANG_TO_BOHR

    @property
    def dv_au(self):
        """in [au]"""
        return self.cell / self.cell_n

    @property
    def x_arr_au(self):
        """in [au]"""
        return np.arange(
            self.origin[0],
            self.origin[0] + (self.cell_n[0] - 0.5) * self.dv_au[0, 0],
            self.dv_au[0, 0],
        )

    @property
    def y_arr_au(self):
        """in [au]"""
        return np.arange(
            self.origin[1],
            self.origin[1] + (self.cell_n[1] - 0.5) * self.dv_au[1, 1],
            self.dv_au[1, 1],
        )

    @property
    def z_arr_au(self):
        """in [au]"""
        return np.arange(
            self.origin[2],
            self.origin[2] + (self.cell_n[2] - 0.5) * self.dv_au[2, 2],
            self.dv_au[2, 2],
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
