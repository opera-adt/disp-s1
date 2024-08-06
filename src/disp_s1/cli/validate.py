import click
from dolphin import setup_logging

from disp_s1.validate import DSET_DEFAULT, compare


@click.command()
@click.argument("golden", type=click.Path(exists=True))
@click.argument("test", type=click.Path(exists=True))
@click.option("--data-dset", default=DSET_DEFAULT)
@click.option("--debug", is_flag=True)
def validate(golden: str, test: str, data_dset: str, debug: bool) -> None:
    """Compare two HDF4 files for consistency."""
    setup_logging(logger_name="disp_s1", debug=debug, filename=None)
    compare(golden, test, data_dset)
