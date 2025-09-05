import typer
from ..version import __version__
from pathlib import Path
from ..cube import Cube
import numpy as np
app = typer.Typer(help="Cubehandler: a tool to handle cube files.")


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
):
    """Shrink a cube file or all cube files in a directory."""

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
    elif inp.is_dir():
        out.mkdir(exist_ok=True)
        for file in inp.glob("*cube"):
            out_file = out / (prefix + file.name)
            run_reduction(file.absolute(), out_file)
            
@app.command(name="sum", help="Create a cube as a linear combination of input cubes. Uses atoms/grid from the first cube.")
def sum_cubes(
    sum_spec: str = typer.Option(
        ...,
        "--files-and-weights",
        help=(
            "Comma-separated list: first all input cube files, then the same number of weights. "
            'Example: "a.cube,b.cube,1.0,2.5" (2 files, 2 weights).'
        ),
    ),
    output_path: str = typer.Argument(..., help="Path to the output cube file."),
    low_precision: bool = typer.Option(
        True,
        "--low-precision/--no-low-precision",
        help="Use low precision for the output cube.",
    ),
):
    """
    Compute: result = w1*data(file1) + w2*data(file2) + ...
    The output cube takes atomic positions / cell / grid from file1.
    """

    # Parse spec: first half = files, second half = weights
    tokens = [t.strip() for t in sum_spec.split(",") if t.strip()]
    if len(tokens) < 2 or len(tokens) % 2 != 0:
        raise typer.BadParameter(
            "Invalid --sum format. Provide an even number of comma-separated items: "
            "first N files, then N weights. Example: a.cube,b.cube,1.0,2.5"
        )

    n = len(tokens) // 2
    file_tokens = tokens[:n]
    weight_tokens = tokens[n:]

    files = [Path(p) for p in file_tokens]
    weights = []
    try:
        weights = [float(w) for w in weight_tokens]
    except ValueError as e:
        raise typer.BadParameter(f"Could not parse weights as floats: {e}")

    if len(files) != len(weights):
        raise typer.BadParameter("The number of files must equal the number of weights.")

    # Load first cube (reference for atoms/grid/metadata)
    if not files[0].is_file():
        raise typer.BadParameter(f"Input file not found: {files[0]}")
    ref = Cube.from_file(files[0])

    # Start with weighted data from first file
    try:
        result_data = weights[0] * np.array(ref.data)
    except AttributeError:
        raise typer.BadParameter(
            "Cube object missing `.data` attribute. "
            "Adjust the code to match your Cube API (e.g., `cube.values`)."
        )

    # Add remaining files
    for f, w in zip(files[1:], weights[1:]):
        if not f.is_file():
            raise typer.BadParameter(f"Input file not found: {f}")
        c = Cube.from_file(f)

        # Basic compatibility checks (adjust attribute names if your API differs)
        if getattr(c, "data", None) is None:
            raise typer.BadParameter(
                f"Cube from {f} has no `.data`. Adjust implementation to your Cube API."
            )
        if np.shape(c.data) != np.shape(ref.data):
            raise typer.BadParameter(
                f"Grid mismatch between {files[0]} and {f}: shapes {np.shape(ref.data)} vs {np.shape(c.data)}"
            )

        # Optionally check origin/cell if available in your Cube class
        # if hasattr(ref, "origin") and hasattr(c, "origin") and not np.allclose(ref.origin, c.origin):
        #     raise typer.BadParameter(f"Origin mismatch between {files[0]} and {f}.")
        # if hasattr(ref, "cell") and hasattr(c, "cell") and not np.allclose(ref.cell, c.cell):
        #     raise typer.BadParameter(f"Cell mismatch between {files[0]} and {f}.")

        result_data = result_data + w * np.array(c.data)

    # Put the combined data back into the reference cube and write it
    ref.data = result_data  # adjust if your API uses a different attribute
    out = Path(output_path)
    
    # Ensure parent directory exists
    out.parent.mkdir(parents=True, exist_ok=True)

    ref.write_cube_file(out, low_precision=low_precision)
