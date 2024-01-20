import click

from disp_s1.create import get_params, make_product


@click.command()
@click.argument(
    "process_dir", type=click.Path(exists=True, dir_okay=True, resolve_path=True)
)
@click.argument("cslc_list", type=click.File("r"))
@click.argument("pair", type=str)
@click.argument("frame_id", type=str)
@click.argument("processing_mode", type=str)
def create(process_dir, cslc_list, pair, frame_id, processing_mode) -> None:
    """Create DISP-S1 product for given pair."""
    make_product(get_params(process_dir, cslc_list, pair, frame_id, processing_mode))
