import click


@click.command()
@click.argument(
    "process_dir", type=click.Path(exists=True, dir_okay=True, resolve_path=True)
)
@click.argument("cslc_list", type=click.File("r"))
@click.argument("pair", type=str)
def create(process_dir, cslc_list, pair):
    """Create DISP-S1 product for given pair."""
    from disp_s1.product import create_output_product
    from pathlib import Path
    from disp_s1.pge_runconfig import RunConfig

    # Get necessary files.
    prod_dict = {}
    prod_dict["unw_filename"] = next(
        Path(f"{process_dir}/unwrapped/").glob(f"{pair}.unw.tif")
    )
    prod_dict["conncomp_filename"] = next(
        Path(f"{process_dir}/unwrapped/").glob(f"{pair}.unw.conncomp")
    )
    prod_dict["ifg_corr_filename"] = next(
        Path(f"{process_dir}/interferograms/").glob(f"{pair}.cor")
    )

    prod_dict["tcorr_filename"] = next(
        Path(f"{process_dir}/interferograms/").glob("temporal_coherence*tif")
    )
    prod_dict["ps_mask_filename"] = next(
        Path(f"{process_dir}/interferograms/").glob("ps_mask_looked.tif")
    )
    prod_dict["pge_runconfig"] = next(Path(f"{process_dir}/").glob("*.yaml"))
    cslc_files = sorted(cslc_list.read().splitlines())
    prod_dict["cslc_files"] = cslc_files

    create_output_product(
        unw_filename=prod_dict["unw_filename"],
        conncomp_filename=prod_dict["conncomp_filename"],
        tcorr_filename=prod_dict["tcorr_filename"],
        ifg_corr_filename=prod_dict["ifg_corr_filename"],
        output_name=f"{pair}.nc",
        corrections={},
        ps_mask_filename=prod_dict["ps_mask_filename"],
        pge_runconfig=RunConfig.from_yaml(prod_dict["pge_runconfig"]),
        cslc_files=prod_dict["cslc_files"],
    )
