import click

from disp_s1.validate import DSET_DEFAULT, compare


@click.command()
@click.argument("golden", type=click.Path(exists=True))
@click.argument("test", type=click.Path(exists=True))
@click.option("--data-dset", default=DSET_DEFAULT)
def validate(golden: str, test: str, data_dset: str) -> None:
    """Compare two HDF4 files for consistency."""
    compare(golden, test, data_dset)
