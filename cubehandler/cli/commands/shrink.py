import click

from ...cube import Cube


@click.command(help="Shrink a cube file.")
@click.argument("input_cube", type=click.Path(exists=True))
@click.argument("output_cube", type=click.Path())
def shrink(input_cube, output_cube):
    cube = Cube.from_file(input_cube)
    cube.reduce_data_density(points_per_angstrom=2)
    cube.rescale_data()
    cube.write_cube_file(output_cube, low_precision=True)
