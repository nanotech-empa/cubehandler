"""The renderer CLI passes the selected output settings unchanged."""

from pathlib import Path

from typer.testing import CliRunner
from cubehandler import Cube
from cubehandler.cli.main import app


def test_render_cli_settings(monkeypatch, tmp_path):
    requests = []

    def render(self, spec):
        requests.append(spec)
        return spec.output_path

    monkeypatch.setattr(Cube, "render", render)
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
