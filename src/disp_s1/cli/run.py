import click

__all__ = ["run_cli", "run_main"]


def run_main(config_file: str, debug: bool = False) -> None:
    """Run the displacement workflow for CONFIG_FILE."""
    # rest of imports here so --help doesn't take forever
    from disp_s1.main import run
    from disp_s1.pge_runconfig import RunConfig

    pge_runconfig = RunConfig.from_yaml(config_file)
    cfg = pge_runconfig.to_workflow()
    run(cfg, pge_runconfig=pge_runconfig, debug=debug)


@click.command("run")
@click.argument("config_file", type=click.Path(exists=True))
@click.pass_context
def run_cli(
    ctx: click.Context,
    config_file: str,
) -> None:
    """Run the displacement workflow for CONFIG_FILE."""
    run_main(config_file=config_file, debug=ctx.obj["debug"])
