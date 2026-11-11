"""Exercise the real YAML/CLI argument contract without rendering or science."""

from pathlib import Path

import numpy as np
import yaml
from ase import Atoms
from typer.testing import CliRunner

from cubehandler import Cube
from cubehandler.cli.main import app


def cube_file(tmp_path):
    path = tmp_path / "source.cube"
    Cube(
        title="Synthetic fixture",
        comment="Invented cube grid",
        ase_atoms=Atoms("H", cell=[4, 4, 4]),
        origin=np.zeros(3),
        cell=np.eye(3) * 8,
        data=np.arange(64).reshape(4, 4, 4) * 0.01,
    ).write_cube_file(path)
    return path


def test_yaml_render_tuple_and_repeated_isosurfaces(tmp_path, monkeypatch):
    source = cube_file(tmp_path)
    records = []

    def render(cube, spec):
        records.append(spec)
        return Path(spec.output_path)

    monkeypatch.setattr(Cube, "render", render)
    recipe = tmp_path / "recipe.yml"
    recipe.write_text(
        yaml.safe_dump(
            {
                "steps": [
                    {
                        "command": "render",
                        "args": [str(source)],
                        "options": {
                            "orientation": np.eye(4).ravel().tolist(),
                            "iso": [".1:#FF0000", "-.1:#0000FF"],
                            "format": "png",
                            "output": str(tmp_path / "view.png"),
                            "opacity": 0.35,
                        },
                    }
                ]
            }
        )
    )
    result = CliRunner().invoke(app, ["run", str(recipe)])
    assert result.exit_code == 0, result.output
    assert len(records) == 1
    assert records[0].surface_opacity == 0.35
    assert len(records[0].iso_pairs) == 2


def test_yaml_shrink_boolean_flag_has_no_spurious_input(tmp_path):
    source = cube_file(tmp_path)
    recipe = tmp_path / "recipe.yml"
    output = tmp_path / "reduced"
    recipe.write_text(
        yaml.safe_dump(
            {
                "steps": [
                    {
                        "command": "shrink",
                        "args": [str(source)],
                        "options": {"low_precision": True, "output_dir": str(output)},
                    }
                ]
            }
        )
    )
    result = CliRunner().invoke(app, ["run", str(recipe)])
    assert result.exit_code == 0, result.output
    assert (output / "reduced_source.cube").is_file()
    assert "--low-precision True" not in result.output


def test_yaml_render_failure_propagates_and_stops_next_step(tmp_path):
    recipe = tmp_path / "recipe.yml"
    recipe.write_text(
        yaml.safe_dump(
            {
                "steps": [
                    {
                        "command": "render",
                        "args": ["missing.cube"],
                        "options": {
                            "orientation": np.eye(4).ravel().tolist(),
                            "iso": [".1:#FF0000"],
                            "format": "png",
                        },
                    },
                    {"command": "sum", "args": []},
                ]
            }
        )
    )
    result = CliRunner().invoke(app, ["run", str(recipe)])
    assert result.exit_code != 0
    assert "Running command: sum" not in result.output
