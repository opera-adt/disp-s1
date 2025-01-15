import click

from .dem import stage_dem
from .download import download_group
from .make_browse import make_browse
from .run import run_cli
from .validate import validate


@click.group(name="disp-s1")
@click.version_option()
@click.option("--debug", is_flag=True, help="Add debug messages to the log.")
@click.pass_context
def cli_app(ctx: click.Context, debug: bool) -> None:
    """Run a displacement workflow."""
    # https://click.palletsprojects.com/en/8.1.x/commands/#nested-handling-and-contexts
    ctx.ensure_object(dict)
    ctx.obj["debug"] = debug


cli_app.add_command(run_cli)
cli_app.add_command(validate)
cli_app.add_command(make_browse)
cli_app.add_command(download_group)
cli_app.add_command(stage_dem)

if __name__ == "__main__":
    cli_app()
