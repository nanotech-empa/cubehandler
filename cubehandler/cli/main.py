import enum
import re
from pathlib import Path
from typing import Optional

import typer
import yaml

from ..cube import Cube, RenderSpec, SUPPORTED_IMAGE_FORMATS
from ..version import __version__

app = typer.Typer(help="Cubehandler: a tool to handle cube files.")
ISO_PATTERN = re.compile(
    r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*:\s*(#[0-9A-Fa-f]{6})\s*$"
)
Orientation16 = tuple[
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
]


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


@app.command(help="Shrink a cube file.")
def shrink(
    input_path: str = typer.Argument(..., help="Path to the input file or directory."),
    output_path: str = typer.Argument(
        ..., help="Path to the output file or directory."
    ),
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
    inp = Path(input_path)
    out = Path(output_path)

    def run_reduction(inp, out):
        cube = Cube.from_file(inp)
        if method == "skimage":
            cube.reduce_data_density_skimage()
        elif method == "slicing":
            cube.reduce_data_density_slicing()
        else:
            raise ValueError(f"Unknown method: {method}")

        cube.write_cube_file(out, low_precision=low_precision)

    if inp.is_file():
        run_reduction(inp, out)
        output["shrink"].append(
            {
                "input": str(inp),
                "output": str(out),
                "method": method,
                "low_precision": low_precision,
            }
        )
    elif inp.is_dir():
        out.mkdir(exist_ok=True)
        for file in inp.glob("*cube"):
            out_file = out / (prefix + file.name)
            run_reduction(file.absolute(), out_file)
            output["shrink"].append(
                {
                    "input": str(inp / file.name),
                    "output": str(out_file),
                    "method": method,
                    "low_precision": low_precision,
                }
            )

    if verbosity >= Verbosity.INFO:
        typer.echo(yaml.dump(output, sort_keys=False))


@app.command(help="Render a cube file.")
def render(
    cube_file: str = typer.Argument(..., help="Path to the input cube file."),
    orientation: Orientation16 = typer.Option(
        ...,
        "--orientation",
        help="Camera orientation matrix from nglview (16 numbers).",
    ),
    iso: list[str] = typer.Option(
        ...,
        "--iso",
        help="Isovalue/color pair in the form <value:#RRGGBB>. Repeat for multiple pairs.",
    ),
    image_format: str = typer.Option(
        ...,
        "--format",
        help=f"Output image format. Supported: {', '.join(SUPPORTED_IMAGE_FORMATS)}.",
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output image path. Defaults to <cube_stem>_render.<format>.",
    ),
):
    """Render a cube file with nglview-compatible camera orientation."""
    input_path = Path(cube_file)
    if not input_path.is_file():
        raise typer.BadParameter(
            f"Cube file does not exist: {input_path}", param_hint="cube_file"
        )

    normalized_format = image_format.lower()
    if normalized_format not in SUPPORTED_IMAGE_FORMATS:
        raise typer.BadParameter(
            f"Unsupported format '{image_format}'. "
            f"Choose one of: {', '.join(SUPPORTED_IMAGE_FORMATS)}.",
            param_hint="--format",
        )

    iso_pairs: list[tuple[float, str]] = []
    for iso_item in iso:
        match = ISO_PATTERN.match(iso_item)
        if match is None:
            raise typer.BadParameter(
                f"Invalid --iso value '{iso_item}'. Use <value:#RRGGBB>.",
                param_hint="--iso",
            )
        isovalue = float(match.group(1))
        color = match.group(2).upper()
        iso_pairs.append((isovalue, color))

    if output is None:
        output_path = Path.cwd() / f"{input_path.stem}_render.{normalized_format}"
    else:
        output_path = Path(output)
        if not output_path.suffix:
            output_path = output_path.with_suffix(f".{normalized_format}")
        elif output_path.suffix.lower() != f".{normalized_format}":
            raise typer.BadParameter(
                f"Output extension '{output_path.suffix}' does not match --format '{normalized_format}'.",
                param_hint="--output",
            )

    cube = Cube.from_file(input_path)
    render_spec = RenderSpec(
        orientation16=tuple(float(v) for v in orientation),
        iso_pairs=tuple(iso_pairs),
        image_format=normalized_format,
        output_path=output_path,
    )

    try:
        rendered_file = cube.render(render_spec)
    except (ImportError, RuntimeError, ValueError) as exc:
        raise typer.BadParameter(str(exc))

    typer.echo(f"Rendered cube file: {rendered_file}")
