import click

from .run import run
from .validate import validate


@click.group()
@click.version_option()
@click.option("--debug", is_flag=True, help="Add debug messages to the log.")
@click.pass_context
def cli_app(ctx: click.Context, debug: bool) -> None:
    """Run a displacement workflow."""
    # https://click.palletsprojects.com/en/8.1.x/commands/#nested-handling-and-contexts
    ctx.ensure_object(dict)
    ctx.obj["debug"] = debug


cli_app.add_command(run)
cli_app.add_command(validate)

if __name__ == "__main__":
    cli_app()
