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
@click.option(
    "--method",
    default="default",
    help="Method to use for data reduction. Available methods are: slicing, skimiage.",
)
@click.option(
    "--low-precision/--no-low-precision",
    default=True,
    help="Use low precision for the output cube.",
)
def shrink(input_path, output_path, prefix, method="default", low_precision=True):
    inp = Path(input_path)
    out = Path(output_path)

    def run_reduction(inp, out):
        cube = Cube.from_file(inp)
        match method:
            case "skimage":
                cube.reduce_data_density_skimage()
            case "default":
                cube.reduce_data_density_slicing()
            case _:
                raise ValueError(f"Unknown method: {method}")

        cube.write_cube_file(out, low_precision=low_precision)

    if inp.is_file():
        run_reduction(inp, out)
    elif inp.is_dir():
        out.mkdir(exist_ok=True)
        for file in inp.glob("*cube"):
            out_file = out / (prefix + file.name)
            run_reduction(file.absolute(), out_file)
