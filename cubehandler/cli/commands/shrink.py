from pathlib import Path

import click

from ...cube import Cube


@click.command(help="Shrink a cube file.")
@click.argument(
    "input_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=True),
    required=True,
)
@click.argument(
    "output_path", type=click.Path(file_okay=True, dir_okay=True), required=True
)
@click.option("-p", "--prefix", default="reduced_")
def shrink(input_path, output_path, prefix):
    inp = Path(input_path)
    out = Path(output_path)

    def run_reduction(inp, out):
        cube = Cube.from_file(inp)
        # cube.reduce_data_density(points_per_angstrom=2)
        print("Reducing data density...")
        cube.reduce_data_density_skimage()
        # cube.rescale_data() # Rescaling happens below when low_precision is True
        cube.write_cube_file(out, low_precision=True)

    if inp.is_file():
        run_reduction(inp, out)
    elif inp.is_dir():
        out.mkdir(exist_ok=True)
        for file in inp.glob("*cube"):
            out_file = out / (prefix + file.name)
            run_reduction(file.absolute(), out_file)
