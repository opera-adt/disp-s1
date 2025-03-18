from pathlib import Path

import click

__all__ = ["run_cli", "run_main"]


def run_main(config_file: str, debug: bool = False) -> None:
    """Run the displacement workflow for CONFIG_FILE."""
    # rest of imports here so --help doesn't take forever
    from disp_s1.pge_runconfig import RunConfig, StaticLayersRunConfig

    if _is_static_layers_workflow(config_file):
        from disp_s1.main_static_layers import run_static_layers

        pge_runconfig = StaticLayersRunConfig.from_yaml(config_file)

        return run_static_layers(pge_runconfig=pge_runconfig)

    from disp_s1.main import run

    pge_runconfig = RunConfig.from_yaml(config_file)
    cfg = pge_runconfig.to_workflow()
    run(cfg, pge_runconfig=pge_runconfig, debug=debug)


def _is_static_layers_workflow(yaml_path: Path | str) -> bool:
    """Check if the workflow is a static layers workflow."""
    from ruamel.yaml import YAML

    y = YAML(typ="safe")
    with open(yaml_path) as f:
        data = y.load(f)
    return data["primary_executable"]["product_type"] == "DISP_S1_STATIC"


@click.command("run")
@click.argument("config_file", type=click.Path(exists=True))
@click.pass_context
def run_cli(
    ctx: click.Context,
    config_file: str,
) -> None:
    """Run the displacement workflow for CONFIG_FILE."""
    run_main(config_file=config_file, debug=ctx.obj["debug"])
