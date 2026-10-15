"""The renderer CLI passes the selected output settings unchanged."""

from pathlib import Path
import json
import pytest

from typer.testing import CliRunner
from cubehandler import Cube
from cubehandler.cli.main import app


@pytest.mark.parametrize("with_plane", [False, True])
def test_render_cli_settings(monkeypatch, tmp_path, with_plane):
    requests = []

    def render(self, spec):
        requests.append(spec)
        return spec.output_path

    monkeypatch.setattr(Cube, "render", render)
    extra = []
    if with_plane:
        path = tmp_path / "plane.json"
        path.write_text(
            json.dumps(
                dict(
                    center=[0, 0, 0],
                    normal=[0, 0, 1],
                    offset=0.2,
                    width=3.0,
                    height=4.0,
                    resolution=64,
                    vmin=-0.1,
                    vmax=0.1,
                    periodic=False,
                )
            )
        )
        extra = ["--slice-plane", str(path)]
    result = CliRunner().invoke(
        app,
        [
            "render",
            str(Path(__file__).parent / "CH4_HOMO.cube"),
            "--orientation",
            *map(str, [20, 0, 0, 0, 0, 20, 0, 0, 0, 0, 20, 0, 0, 0, 0, 1]),
            "--iso",
            "0.03:#2563B7",
            "--iso",
            "-0.03:#E3AE3B",
            "--format",
            "png",
            "--output",
            str(tmp_path / "render.png"),
            "--width",
            "640",
            "--height",
            "480",
            "--background",
            "#123456",
            "--opacity",
            ".7",
            *extra,
        ],
    )
    assert result.exit_code == 0, result.output
    spec = requests[0]
    assert (spec.width, spec.height, spec.background, spec.surface_opacity) == (
        640,
        480,
        "#123456",
        0.7,
    )
    assert spec.iso_pairs == ((0.03, "#2563B7"), (-0.03, "#E3AE3B"))
    assert (spec.slice_plane is not None) == with_plane
    if with_plane:
        assert spec.slice_plane.offset == 0.2 and spec.slice_plane.periodic is False
