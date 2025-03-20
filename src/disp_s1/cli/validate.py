from pathlib import Path

import click

from disp_s1.validate import DSET_DEFAULT, compare, compare_static_layers


@click.command()
@click.argument("golden", type=click.Path(exists=True))
@click.argument("test", type=click.Path(exists=True))
@click.option("--data-dset", default=DSET_DEFAULT)
@click.option("--debug", is_flag=True)
def validate(golden: str, test: str, data_dset: str, debug: bool) -> None:
    """Validate an OPERA DISP-S1 product."""
    from dolphin import setup_logging

    setup_logging(logger_name="disp_s1", debug=debug, filename=None)
    compare(golden, test, data_dset)


@click.command()
@click.argument("golden", type=click.Path(exists=True))
@click.argument("test", type=click.Path(exists=True))
@click.option("--debug", is_flag=True)
def validate_static_layers(golden: str, test: str, debug: bool) -> None:
    """Validate the OPERA DISP-S1-STATIC products."""
    from dolphin import setup_logging

    setup_logging(logger_name="disp_s1", debug=debug, filename=None)
    compare_static_layers(Path(golden), Path(test))
