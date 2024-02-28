import click

from disp_s1.validate import DSET_DEFAULT
from disp_s1.validate import validate as _validate


@click.command()
@click.argument("test", type=click.Path(exists=True))
@click.option("--golden", type=click.Path(exists=True))
@click.option("--igram", type=click.Path(exists=True))
@click.option("--json", type=click.Path(exists=True))
@click.option("--data-dset", default=DSET_DEFAULT)
def validate(
    test: str, golden: str | None, igram: str | None, json: str | None, data_dset: str
) -> None:
    """Validate an OPERA DISP-S1 product."""
    _validate(
        test, golden_file=golden, igram_file=igram, json_file=json, data_dset=data_dset
    )
