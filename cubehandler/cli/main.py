import typer
from pathlib import Path
from ..cube import Cube
import enum
import yaml

from ..version import __version__

app = typer.Typer(help="Cubehandler: a tool to handle cube files.")


class Verbosity(enum.IntEnum):
    """Verbosity levels."""

    ERROR = 0
    WARNING = 1
    INFO = 2
    DEBUG = 3


# Define the shared verbosity option
VerbosityOption = typer.Option(
    0, "--verbose", "-v", count=True, help="Increase output verbosity (-v, -vv, -vvv)"
)


@app.callback()
def main(
    version: bool = typer.Option(
        None,
        "--version",
        callback=lambda value: typer.echo(f"cubehandler version {__version__}")
        if value
        else None,
        is_eager=True,
        help="Show the version and exit.",
    ),
):
    """Cubehandler: a tool to handle cube files."""
    pass


def input_paths_argument():
    return typer.Argument(
        ..., exists=True, readable=True, help="One or more cube files to process."
    )


def output_dir_option():
    return typer.Option(
        None,
        "--output-dir",
        "-d",
        file_okay=False,
        dir_okay=True,
        writable=True,
        exists=False,
        help="Directory to place output files (default: next to inputs).",
    )


@app.command(help="Shrink a cube file.")
def shrink(
    input_paths: list[Path] = input_paths_argument(),
    output_dir: Path = output_dir_option(),
    prefix: str = typer.Option(
        "reduced_", "--prefix", "-p", help="Prefix for output files."
    ),
    method: str = typer.Option(
        "slicing",
        "--method",
        help="Method to use for data reduction. Available methods are: slicing, skimage.",
    ),
    low_precision: bool = typer.Option(
        True,
        "--low-precision/--no-low-precision",
        help="Use low precision for the output cube.",
    ),
    verbosity: int = VerbosityOption,
):
    """Shrink a cube file or all cube files in a directory."""

    output = {"shrink": []}

    def run_reduction(inp, out):
        cube = Cube.from_file(inp)
        if method == "skimage":
            cube.reduce_data_density_skimage()
        elif method == "slicing":
            cube.reduce_data_density_slicing()
        else:
            raise ValueError(f"Unknown method: {method}")

        cube.write_cube_file(out, low_precision=low_precision)

    # Create output directory if it doesn't exist
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    for inp in input_paths:
        if output_dir is None:
            out_file = inp.parent / (prefix + inp.name)
        else:
            out_file = output_dir / (prefix + inp.name)
        run_reduction(inp, out_file)
        output["shrink"].append(
            {
                "input": str(inp),
                "output": str(out_file),
                "method": method,
                "low_precision": low_precision,
            }
        )

    if verbosity >= Verbosity.INFO:
        typer.echo(yaml.dump(output, sort_keys=False))


@app.command(help="Render a cube file.")
def render(input_path: str = typer.Argument(..., help="Path to the input cube file.")):
    """Render a cube file."""
    inp = Path(input_path)

    def run_render(inp):
        cube = Cube.from_file(inp)
        cube.render()

    if inp.is_file():
        run_render(inp)
        typer.echo(f"Rendering cube file: {inp}")
    else:
        typer.echo("Please provide a valid cube file.")
