from pathlib import Path

import click

from ...cube import Cube


@click.command(help="Render a cube file.")
@click.argument(
    "input_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=True),
    required=True,
)
def render(input_path):
    inp = Path(input_path)

    def run_render(inp):
        cube = Cube.from_file(inp)
        print("RUNNING RENDER (to be implemented)")
        cube.render()

    if inp.is_file():
        run_render(inp)
    else:
        click.echo("Please provide a valid cube file.")
        return
    click.echo(f"Rendering cube file: {inp}")
