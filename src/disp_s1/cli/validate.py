import click
from dolphin import setup_logging

from disp_s1.validate import DSET_DEFAULT
from disp_s1.validate import validate as _validate


@click.command()
@click.argument("test", type=click.Path(exists=True))
@click.option("--golden", type=click.Path(exists=True))
@click.option("--igram", type=click.Path(exists=True))
@click.option("--json", type=click.Path(exists=True))
@click.option("--data-dset", default=DSET_DEFAULT)
@click.option("--debug", is_flag=True)
def validate(
    test: str, golden: str | None, igram: str | None, json: str | None, data_dset: str, debug: bool
) -> None:
    """Validate an OPERA DISP-S1 product."""
    setup_logging(logger_name="disp_s1", debug=debug, filename=None)
    _validate(
        test, golden_file=golden, igram_file=igram, json_file=json, data_dset=data_dset
    )
