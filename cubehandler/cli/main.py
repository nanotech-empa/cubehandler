import click

from ..version import __version__
from .commands import shrink


@click.group(help="Cubehandler: a tool to handle cube files.")
@click.version_option(
    __version__, package_name="cubehandler", message="cubehandler version %(version)s"
)
def cli():
    pass


cli.add_command(shrink.shrink)
