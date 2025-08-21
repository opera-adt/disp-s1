import argparse
import shutil
import subprocess
from collections.abc import Iterable
from itertools import chain, islice, zip_longest
from pathlib import Path
from typing import Iterator, Literal, TypeVar

import opera_utils.geometry

import disp_s1.main
from disp_s1 import pge_runconfig

T = TypeVar("T")


def batched(iterable: Iterable[T], n: int) -> Iterator[tuple[T, ...]]:
    if n < 1:
        raise ValueError("n must be at least one")
    iterator = iter(iterable)
    while batch := tuple(islice(iterator, n)):
        yield batch


def create_batches(burst_to_cslc: dict[str, list[Path]], ms_size: int = 15):
    """Batch real CSLCs per burst (historical mode)."""
    burst_ids = sorted(burst_to_cslc.keys())

    burst_batches = []
    for burst_id in burst_ids:
        cslc_files = sorted(burst_to_cslc[burst_id])
        burst_batches.append(
            [cslc_files[i : i + ms_size] for i in range(0, len(cslc_files), ms_size)]
        )

    combined_batches = []
    for batch_group in zip_longest(*burst_batches, fillvalue=[]):
        combined_batch = list(chain.from_iterable(batch_group))
        if combined_batch:
            combined_batches.append(combined_batch)

    return combined_batches


def get_forward_batch(
    run_idx: int, burst_to_cslc: dict[str, list[Path]], ms_size: int = 15
):
    """Batch real CSLCs per burst (historical mode)."""
    burst_ids = sorted(burst_to_cslc.keys())

    combined_files = []
    for burst_id in burst_ids:
        cslc_files = sorted(burst_to_cslc[burst_id])
        combined_files.extend(cslc_files[run_idx : run_idx + ms_size + 1])

    return combined_files


def latest_k_per_burst(burst_map: dict[str, list[Path]], k: int) -> list[Path]:
    """Return the latest k items per-burst (sorted per burst)."""
    out: list[Path] = []
    for bid in sorted(burst_map.keys()):
        files = sorted(burst_map[bid])
        if not files:
            continue
        out.extend(files[-k:])
    return out


def write_runconfig_from_template(
    template_path: Path | str,
    output_path: Path | str,
    *,
    mode: Literal["DISP_S1_HISTORICAL", "DISP_S1_FORWARD"],
    batch_dir: str,
    frame_id: int,
    save_compressed_slc: bool = False,
) -> None:
    with open(template_path) as f_in, open(output_path, "w") as f_out:
        y = f_in.read()
        # Ensure template has {batch_dir}, {frame_id}, {mode}, {save_compressed_slc}
        f_out.write(
            y.format(
                batch_dir=batch_dir,
                frame_id=frame_id,
                mode=mode,
                save_compressed_slc=str(save_compressed_slc).lower(),
            )
        )


def symlink_inputs(cur_slcs: list[Path], dest_dir: Path) -> None:
    dest_dir.mkdir(exist_ok=True, parents=True)
    for f in cur_slcs:
        new_symlink = (dest_dir / f.name).resolve()
        if not new_symlink.exists():
            new_symlink.symlink_to(f.resolve())


def run_local(runconfig_file: Path) -> None:
    script_name = Path(__file__).parent / "rename_output.py"
    rc = pge_runconfig.RunConfig.from_yaml(runconfig_file)
    cfg = rc.to_workflow()
    cfg.worker_settings.gpu_enabled = True
    disp_s1.main.run(cfg, rc)
    output_dir = rc.product_path_group.output_directory
    comp_ouput_dir = output_dir / "compressed_slcs"
    try:
        subprocess.run(f"{script_name} {output_dir}/*")
        subprocess.run(f"{script_name} {comp_ouput_dir}/*")
    except Exception as e:
        print(e)


def _run_docker(docker_tag: str, runconfig_file: Path | str) -> None:
    cmd = (
        "docker run --rm -u $(id -u):$(id -g) --cpuset-cpus 0-31 --memory '64G' "
        f"-v $PWD:/work {docker_tag} disp-s1 run {runconfig_file}"
    )
    print("Running docker:\n", cmd)
    subprocess.run(cmd, shell=True, check=True)


def _move_newest_file(src_root: Path, dst_dir: Path) -> Path | None:
    """Find the single most recently modified file underneath src_root (recursively)
    that is a regular file, and move it to dst_dir. Returns the new path or None.
    """
    candidates = [p for p in src_root.rglob("*") if p.is_file()]
    if not candidates:
        return None
    newest = max(candidates, key=lambda p: p.stat().st_mtime)
    dst_dir.mkdir(parents=True, exist_ok=True)
    new_path = dst_dir / newest.name
    shutil.move(str(newest), str(new_path))
    return new_path


def move_compressed_to_bulk(run_output_dir: Path, bulk_dir: Path) -> int:
    """Move all files under output/compressed_slcs into a bulk directory.

    Returns the number of files moved.
    """
    comp_dir = run_output_dir / "compressed_slcs"
    if not comp_dir.exists():
        return 0
    files = [p for p in comp_dir.glob("*") if p.is_file()]
    if not files:
        return 0
    bulk_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for f in files:
        shutil.move(str(f), str(bulk_dir / f.name))
        count += 1
    return count


def run_once_historical(
    *,
    run_idx: int,
    frame_id: int,
    ms_size: int,
    num_compressed: int,
    docker_tag: str | None,
    template_path: Path,
) -> None:
    """Run one minsitack in historical mode."""
    batch_str = f"batch-{run_idx:02d}"
    batch_dir = Path(batch_str)
    batch_dir.mkdir(exist_ok=True)

    runconfig_file = Path(f"runconfig_historical_{batch_str}.yaml")

    all_cslc_files = sorted(Path("input_slcs").resolve().glob("*.h5"))
    burst_to_cslc = opera_utils.group_by_burst(all_cslc_files)
    batches = create_batches(burst_to_cslc, ms_size=ms_size)

    cur_real_slcs = list(batches[run_idx])

    all_compressed_files = sorted(
        Path().glob("batch-*/output/compressed_slcs/compressed_*.h5")
    )
    burst_to_compressed = opera_utils.group_by_burst(all_compressed_files)
    compressed_files = []
    if burst_to_compressed:
        burst_ids = sorted(burst_to_cslc.keys())
        for burst_id in burst_ids:
            # Add the last K compressed
            compressed_files.extend(burst_to_compressed[burst_id][-num_compressed:])

    cur_slcs = compressed_files + cur_real_slcs

    symlink_inputs(cur_slcs, batch_dir / "input_slcs")
    write_runconfig_from_template(
        template_path,
        runconfig_file,
        batch_dir=batch_str,
        frame_id=frame_id,
        mode="DISP_S1_HISTORICAL",
        save_compressed_slc=True,
    )

    if docker_tag:
        _run_docker(docker_tag, runconfig_file)
    else:
        run_local(runconfig_file)


def run_once_forward(
    *,
    run_idx: int,
    frame_id: int,
    ms_size: int,
    num_compressed: int,
    docker_tag: str | None,
    template_path: Path,
    bulk_compressed_dir: Path,
    forward_outputs_dir: Path,
) -> None:
    """Run forward mode.

    - Real CSLCs: latest ms_size per-burst from input_slcs/
    - Compressed CSLCs: latest num_compressed per-burst from bulk_compressed_dir
    - Every ms_size runs: toggle save_compressed_slc=True and migrate
        produced compressed to bulk
    - Always move the single newest product from this run into forward_outputs_dir
    """
    batch_str = f"batch-{run_idx:02d}"
    batch_dir = Path(batch_str)
    input_stage = batch_dir / "input_slcs"
    output_dir = batch_dir / "output"
    batch_dir.mkdir(exist_ok=True)

    # Toggle compress output every ms_size runs (1-based cadence)
    save_compressed_now = (run_idx + 1) % ms_size == 0

    runconfig_file = Path(f"runconfig_forward_{batch_str}.yaml")
    write_runconfig_from_template(
        template_path,
        runconfig_file,
        batch_dir=batch_str,
        frame_id=frame_id,
        mode="DISP_S1_FORWARD",
        save_compressed_slc=save_compressed_now,
    )

    # Real CSLCs (latest ms_size per burst)
    all_cslc_files = sorted(Path("input_slcs").resolve().glob("*.h5"))
    burst_to_cslc = opera_utils.group_by_burst(all_cslc_files)
    cur_real_slcs = get_forward_batch(
        run_idx=run_idx, burst_to_cslc=burst_to_cslc, ms_size=ms_size
    )

    # Compressed CSLCs from bulk (latest num_compressed per burst)
    all_compressed_files = sorted(bulk_compressed_dir.glob("*.h5"))
    burst_to_compressed = opera_utils.group_by_burst(all_compressed_files)
    # TODO: decide if we wanna watch for overlap in time...
    latest_comp = (
        latest_k_per_burst(burst_to_compressed, num_compressed)
        if burst_to_compressed
        else []
    )

    # Order: compressed first, then real
    cur_slcs = list(latest_comp) + list(cur_real_slcs)
    symlink_inputs(cur_slcs, input_stage)

    # Run
    if docker_tag:
        _run_docker(docker_tag, runconfig_file)
    else:
        run_local(runconfig_file)

    # Move newest product from this run into forward outputs stash
    moved = _move_newest_file(output_dir, forward_outputs_dir)
    if moved:
        print(f"[forward] staged newest product -> {moved}")

    # If this is a compression run, move produced compressed files into bulk
    if save_compressed_now:
        moved_n = move_compressed_to_bulk(output_dir, bulk_compressed_dir)
        print(
            f"[forward] moved {moved_n} compressed SLC(s) to bulk:"
            f" {bulk_compressed_dir}"
        )


if __name__ == "__main__":
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", type=int, nargs="*", required=True, help="Run indices (0-based)"
    )
    parser.add_argument("--frame-id", type=int, required=True)
    parser.add_argument("--ms-size", type=int, default=15)
    parser.add_argument("--num-compressed", type=int, default=5)
    parser.add_argument("--docker-tag")
    parser.add_argument(
        "--mode",
        choices=["historical", "forward"],
        default="historical",
        help="Test cadence mode",
    )
    parser.add_argument(
        "--template",
        default="runconfig_template.yaml.txt",
        help="Runconfig template with {batch_dir}, {frame_id}, {save_compressed_slc}",
    )
    parser.add_argument(
        "--bulk-compressed-dir",
        type=Path,
        default=Path("compressed_slcs_bulk"),
        help="Folder accumulating all compressed_*.h5 across runs (forward mode)",
    )
    parser.add_argument(
        "--forward-outputs-dir",
        type=Path,
        default=Path("forward_outputs"),
        help="Where to stage the single newest product per forward run",
    )

    # Keep JAX memory small
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", ".10")

    args = parser.parse_args()
    template_path = Path(args.template)

    for run_idx in args.i:
        if args.mode == "historical":
            run_once_historical(
                run_idx=run_idx,
                frame_id=args.frame_id,
                ms_size=args.ms_size,
                num_compressed=args.num_compressed,
                docker_tag=args.docker_tag,
                template_path=template_path,
            )
        else:
            run_once_forward(
                run_idx=run_idx,
                frame_id=args.frame_id,
                ms_size=args.ms_size,
                num_compressed=args.num_compressed,
                docker_tag=args.docker_tag,
                template_path=template_path,
                bulk_compressed_dir=args.bulk_compressed_dir,
                forward_outputs_dir=args.forward_outputs_dir,
            )
