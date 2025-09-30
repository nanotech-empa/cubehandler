import typer
from pathlib import Path
from ..cube import Cube
import enum
import yaml
from glob import glob

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
        ..., exists=False, readable=True, help="One or more cube files to process."
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


def bash_like_globbing(value: Path) -> list[Path]:
    """Expand bash-like globbing patterns in the input path."""
    return [Path(p) for p in glob(str(value))]


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
        for my_path in bash_like_globbing(inp):
            if output_dir is None:
                out_file = my_path.parent / (prefix + my_path.name)
            else:
                out_file = output_dir / (prefix + my_path.name)
            run_reduction(my_path, out_file)
            output["shrink"].append(
                {
                    "input": str(my_path),
                    "output": str(out_file),
                    "method": method,
                    "low_precision": low_precision,
                }
            )

    if verbosity >= Verbosity.INFO:
        typer.echo(yaml.dump(output, sort_keys=False))


def parse_pairs(pairs: list[str]) -> list[tuple[Path, float]]:
    if len(pairs) % 2 != 0:
        typer.echo("Error: each cube file must be followed by a coefficient.", err=True)
        raise typer.Exit(1)
    result = []
    for i in range(0, len(pairs), 2):
        path = Path(pairs[i])
        coeff_str = pairs[i + 1]
        try:
            coeff = float(coeff_str)
        except ValueError:
            typer.echo(f"Error: {coeff_str} is not a valid float.", err=True)
            raise typer.Exit(1)
        result.append((path, coeff))
    return result


@app.command()
def sum(
    inputs: list[str] = typer.Argument(
        ..., allow_dash=True, help="Pairs of cube file and coefficient."
    ),
    output: Path = typer.Option(..., "--output", "-o", help="Output cube file."),
    overwrite: bool = typer.Option(
        False, "--overwrite", help="Allow overwriting output file."
    ),
    verbosity: int = VerbosityOption,
):
    """Sum multiple cube files with scaling coefficients into a single cube file.

    Example usage: cubehandler sum cube1.cube 1.0 cube2.cube 2.0 -o output.cube

    In case of negative coefficients, use '--' to separate options from positional arguments:

    cubehandler sum -o output.cube -- cube1.cube 1.0 cube2.cube -1.0
    """

    output_log = {
        "sum": {
            "inputs": [],
            "output": str(output),
        }
    }

    pairs = parse_pairs(inputs)

    if output.exists() and not overwrite:
        typer.echo(f"Error: {output} already exists (use --overwrite).", err=True)
        raise typer.Exit(1)

    cube = Cube.from_file(pairs[0][0]) * pairs[0][1]
    output_log["sum"]["inputs"].append(
        {"file": str(pairs[0][0]), "coefficient": pairs[0][1]}
    )
    for path, coeff in pairs[1:]:
        output_log["sum"]["inputs"].append({"file": str(path), "coefficient": coeff})
        cube += Cube.from_file(path) * coeff
    cube.write_cube_file(output)
    if verbosity >= Verbosity.INFO:
        typer.echo(yaml.dump(output_log, sort_keys=False))


@app.command()
def run(yaml_file: Path = typer.Argument(..., exists=True, readable=True)):
    """Run a series of operations defined in a YAML file."""
    from typer.testing import CliRunner

    runner = CliRunner()

    with open(yaml_file, "r") as f:
        config = yaml.safe_load(f)

    for step in config.get("steps", []):
        command = step.get("command")
        all_tokens = [command] + [str(arg) for arg in step.get("args", [])]
        options = [
            (f"--{k.replace('_', '-')}", v) for k, v in step.get("options", {}).items()
        ]
        all_tokens += [str(item) for opt in options for item in opt if item is not None]
        result = runner.invoke(app, all_tokens)
        typer.echo(result.stdout)
