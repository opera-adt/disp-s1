#!/usr/bin/env python
from pathlib import Path

import h5py
import numpy as np
from dolphin import Filename, get_log, io
from numpy.typing import ArrayLike

from .utils import FRAME_TO_BURST_JSON_FILE, get_frame_bbox

logger = get_log()


DSET_DEFAULT = "unwrapped_phase"


class ValidationError(Exception):
    """Raised when a product fails a validation check."""


class ComparisonError(ValidationError):
    """Exception raised when two datasets do not match."""


def compare_groups(
    golden_group: h5py.Group,
    test_group: h5py.Group,
    pixels_failed_threshold: float = 0.01,
    diff_threshold: float = 1e-5,
) -> None:
    """Compare all datasets in two HDF5 files.

    Parameters
    ----------
    golden_group : h5py.Group
        Path to the golden file.
    test_group : h5py.Group
        Path to the test file to be compared.
    pixels_failed_threshold : float, optional
        The threshold of the percentage of pixels that can fail the comparison.
    diff_threshold : float, optional
        The abs. difference threshold between pixels to consider failing.

    Raises
    ------
    ComparisonError
        If the two files do not match in all datasets.

    """
    # Check if group names match
    if set(golden_group.keys()) != set(test_group.keys()):
        raise ComparisonError(
            f"Group keys do not match: {set(golden_group.keys())} vs"
            f" {set(test_group.keys())}"
        )

    for key in golden_group.keys():
        if isinstance(golden_group[key], h5py.Group):
            compare_groups(
                golden_group[key],
                test_group[key],
                pixels_failed_threshold,
                diff_threshold,
            )
        else:
            test_dataset = test_group[key]
            golden_dataset = golden_group[key]
            _compare_datasets_attr(golden_dataset, test_dataset)

            if key == "connected_component_labels":
                _validate_conncomp_labels(test_dataset, golden_dataset)
            elif key == "unwrapped_phase":
                test_conncomps = test_group["connected_component_labels"]
                golden_conncomps = golden_group["connected_component_labels"]
                _validate_unwrapped_phase(
                    test_dataset,
                    golden_dataset,
                    test_conncomps,
                    golden_conncomps,
                )
            else:
                _validate_dataset(
                    test_dataset,
                    golden_dataset,
                    pixels_failed_threshold,
                    diff_threshold,
                )


def _compare_datasets_attr(
    golden_dataset: h5py.Dataset, test_dataset: h5py.Dataset
) -> None:
    if golden_dataset.name != test_dataset.name:
        raise ComparisonError(
            f"Dataset names do not match: {golden_dataset.name} vs {test_dataset.name}"
        )
    name = golden_dataset.name

    if golden_dataset.shape != test_dataset.shape:
        raise ComparisonError(
            f"{name} shapes do not match: {golden_dataset.shape} vs"
            f" {test_dataset.shape}"
        )

    if golden_dataset.dtype != test_dataset.dtype:
        raise ComparisonError(
            f"{name} dtypes do not match: {golden_dataset.dtype} vs"
            f" {test_dataset.dtype}"
        )

    if golden_dataset.attrs.keys() != test_dataset.attrs.keys():
        raise ComparisonError(
            f"{name} attribute keys do not match: {golden_dataset.attrs.keys()} vs"
            f" {test_dataset.attrs.keys()}"
        )

    for attr_key in golden_dataset.attrs.keys():
        if attr_key in ("REFERENCE_LIST", "DIMENSION_LIST"):
            continue
        val1, val2 = golden_dataset.attrs[attr_key], test_dataset.attrs[attr_key]
        if isinstance(val1, np.ndarray):
            is_equal = np.allclose(val1, val2, equal_nan=True)
        elif isinstance(val1, np.floating) and np.isnan(val1) and np.isnan(val2):
            is_equal = True
        else:
            is_equal = val1 == val2
        if not is_equal:
            raise ComparisonError(
                f"{name} attribute values for key '{attr_key}' do not match: "
                f"{golden_dataset.attrs[attr_key]} vs {test_dataset.attrs[attr_key]}"
            )


def _fmt_ratio(num: int, den: int, digits: int = 3) -> str:
    """Get a string representation of a rational number as a fraction and percent.

    Parameters
    ----------
    num : int
        The numerator.
    den : int
        The denominator.
    digits : int, optional
        Number of decimal digits to use. Defaults to 3.

    Returns
    -------
    str
        A string representation of the input.

    """
    return f"{num}/{den} ({100.0 * num / den:.{digits}f}%)"


def _validate_conncomp_labels(
    test_dataset: h5py.Dataset,
    ref_dataset: h5py.Dataset,
    threshold: float = 0.9,
) -> None:
    """Validate connected component labels from unwrapping.

    Computes a binary mask of nonzero-valued labels in the test and reference datasets,
    and checks the intersection between the two masks. The dataset fails validation if
    the ratio of the intersection area to the reference mask area is below a
    predetermined minimum threshold.

    Parameters
    ----------
    test_dataset : h5py.Dataset
        HDF5 dataset containing connected component labels to be validated.
    ref_dataset : h5py.Dataset
        HDF5 dataset containing connected component labels to use as reference. Must
        have the same shape as `test_dataset`.
    threshold : float, optional
        Minimum allowable intersection area between nonzero-labeled regions in the test
        and reference dataset, as a fraction of the total nonzero-labeled area in the
        reference dataset. Must be in the interval [0, 1]. Defaults to 0.9.

    Raises
    ------
    ComparisonError
        If the intersecting area between the two masks was below the threshold.

    """
    logger.info("Checking connected component labels...")

    if test_dataset.shape != ref_dataset.shape:
        errmsg = (
            "shape mismatch: test dataset and reference dataset must have the same"
            f" shape, got {test_dataset.shape} vs {ref_dataset.shape}"
        )
        raise ComparisonError(errmsg)

    if not (0.0 <= threshold <= 1.0):
        errmsg = f"threshold must be between 0 and 1, got {threshold}"
        raise ValueError(errmsg)

    # Total size of each dataset.
    size = ref_dataset.size

    # Compute binary masks of pixels with nonzero labels in each dataset.
    test_nonzero = np.not_equal(test_dataset, 0)
    ref_nonzero = np.not_equal(ref_dataset, 0)

    # Compute the intersection & union of both masks.
    intersect = test_nonzero & ref_nonzero
    union = test_nonzero | ref_nonzero

    # Compute the total area of each mask.
    test_area = np.sum(test_nonzero)
    ref_area = np.sum(ref_nonzero)
    intersect_area = np.sum(intersect)
    union_area = np.sum(union)

    # Log some statistics about the unwrapped area.
    logger.info(f"Test unwrapped area: {_fmt_ratio(test_area, size)}")
    logger.info(f"Reference unwrapped area: {_fmt_ratio(ref_area, size)}")
    logger.info(f"Intersection/Reference: {_fmt_ratio(intersect_area, ref_area)}")
    logger.info(f"Intersection/Union: {_fmt_ratio(intersect_area, union_area)}")

    # Compute the ratio of intersection area to area in the reference mask.
    ratio = intersect_area / ref_area

    if ratio < threshold:
        errmsg = (
            f"connected component labels dataset {test_dataset.name!r} failed"
            " validation: insufficient area of overlap between test and reference"
            f" nonzero labels ({ratio} < {threshold})"
        )
        raise ComparisonError(errmsg)


def _validate_unwrapped_phase(
    test_dataset: h5py.Dataset,
    ref_dataset: h5py.Dataset,
    test_conncomps: ArrayLike,
    ref_conncomps: ArrayLike,
    nan_threshold: float = 0.01,
    atol: float = 1e-6,
) -> None:
    """Validate unwrapped phase values against a reference dataset.

    Checks that the phase values in the test dataset are congruent with the reference
    dataset -- that is, their values are approximately the same modulo 2pi.

    Parameters
    ----------
    test_dataset : h5py.Dataset
        HDF5 dataset containing unwrapped phase values to be validated.
    ref_dataset : h5py.Dataset
        HDF5 dataset containing unwrapped phase values to use as reference. Must have
        the same shape as `test_dataset`.
    test_conncomps : array_like
        Connected component labels associated with `test_dataset`.
    ref_conncomps : array_like
        Connected component labels associated with `ref_dataset`.
    nan_threshold : float
        Maximum allowable fraction of NaN values among valid pixels (pixels with nonzero
        connected component label). Must be in the interval [0, 1]. Defaults to 0.01.
    atol : float, optional
        Maximum allowable absolute error between the re-wrapped reference and test
        values, in radians. Must be nonnegative. Defaults to 1e-6.

    Raises
    ------
    ValidationError
        If the NaN value count exceeded the specified threshold.
    ComparisonError
        If the two datasets were not congruent within the specified error tolerance.

    """
    logger.info("Checking unwrapped phase...")

    if test_dataset.shape != ref_dataset.shape:
        errmsg = (
            "shape mismatch: test dataset and reference dataset must have the same"
            f" shape, got {test_dataset.shape} vs {ref_dataset.shape}"
        )
        raise ComparisonError(errmsg)

    if (test_dataset.shape != test_conncomps.shape) or (
        ref_dataset.shape != ref_conncomps.shape
    ):
        errmsg = (
            "shape mismatch: unwrapped phase and connected component labels must have"
            " the same shape"
        )
        raise ValidationError(errmsg)

    if not (0.0 <= nan_threshold <= 1.0):
        errmsg = f"nan_threshold must be between 0 and 1, got {nan_threshold}"
        raise ValueError(errmsg)

    # Get a mask of valid pixels (pixels that had nonzero connected component label) in
    # both the test & reference data.
    test_valid_mask = np.not_equal(test_conncomps, 0)
    ref_valid_mask = np.not_equal(ref_conncomps, 0)
    valid_mask = test_valid_mask & ref_valid_mask

    # Get the total valid area in both datasets.
    test_valid_area = np.sum(test_valid_mask)
    ref_valid_area = np.sum(ref_valid_mask)

    # Get a mask of NaN values in either dataset.
    test_nan_mask = np.isnan(test_dataset)
    ref_nan_mask = np.isnan(ref_dataset)
    nan_mask = test_nan_mask | ref_nan_mask

    # Get the total number of NaN values in the valid regions of each dataset.
    test_nan_count = np.sum(test_nan_mask & test_valid_mask)
    ref_nan_count = np.sum(ref_nan_mask & ref_valid_mask)

    # Log some info about the NaN values.
    logger.info(f"Test nan count: {_fmt_ratio(test_nan_count, test_valid_area)}")
    logger.info(f"Reference nan count: {_fmt_ratio(ref_nan_count, ref_valid_area)}")

    # Compute the fraction of NaN values in the valid region.
    test_nan_frac = test_nan_count / test_valid_area

    if test_nan_frac > nan_threshold:
        errmsg = (
            f"unwrapped phase dataset {test_dataset.name!r} failed validation: too"
            f" many nan values ({test_nan_frac} > {nan_threshold})"
        )
        raise ValidationError(errmsg)

    _check_phase_congruence(
        unw=test_dataset, ref=ref_dataset, mask=(valid_mask & ~nan_mask), atol=atol
    )


def _check_phase_congruence(
    unw: ArrayLike,
    ref: ArrayLike,
    mask: ArrayLike | None = None,
    *,
    atol: float = 1e-6,
) -> None:
    """Check unwrapped phase values for congruence with a reference dataset.

    Parameters
    ----------
    unw : array_like
        The unwrapped dataset, with phase values in radians, to validate.
    ref : array_like
        The reference (wrapped or unwrapped) dataset, with phase values in radians, to
        compare against.
    mask : array_like or None, optional
        An optional binary mask of valid phase values. False elements in the mask
        indicate phase values that are missing or invalid. If None, no mask is applied.
        Defaults to None.
    atol : float, optional
        Maximum allowable absolute error between the unwrapped and reference phase
        values (after re-wrapping), in radians. Must be nonnegative. Defaults to 1e-6.

    Raises
    ------
    ComparisonError
        If the two datasets were not congruent within the specified error tolerance.

    """
    if atol < 0.0:
        errmsg = f"atol must be >= 0, got {atol}"
        raise ValueError(errmsg)

    def rewrap(phi: np.ndarray) -> np.ndarray:
        tau = 2.0 * np.pi
        return phi - tau * np.ceil((phi - np.pi) / tau)

    # Compute the difference between the test & reference values and wrap it to the
    # interval (-pi, pi].
    diff = np.subtract(ref, unw)
    wrapped_diff = rewrap(diff)

    # Exclude masked pixels.
    if mask is not None:
        wrapped_diff = wrapped_diff[mask]

    # Log some statistics about the deviation between the test & reference phase.
    abs_wrapped_diff = np.abs(wrapped_diff)
    mean_abs_err = np.mean(abs_wrapped_diff)
    max_abs_err = np.max(abs_wrapped_diff)
    logger.info(f"Mean absolute re-wrapped phase error: {mean_abs_err:.5f} rad")
    logger.info(f"Max absolute re-wrapped phase error: {max_abs_err:.5f} rad")

    noncongruent_count = np.sum(abs_wrapped_diff > atol)
    logger.info(
        "Non-congruent pixel count:"
        f" {_fmt_ratio(noncongruent_count, wrapped_diff.size)}"
    )

    if noncongruent_count != 0:
        errmsg = (
            "unwrapped phase dataset failed validation: phase values were not"
            " congruent with reference dataset"
        )
        raise ComparisonError(errmsg)


def _validate_dataset(
    test_dataset: h5py.Dataset,
    golden_dataset: h5py.Dataset,
    pixels_failed_threshold: float = 0.01,
    diff_threshold: float = 1e-5,
) -> None:
    """Validate a generic dataset.

    Parameters
    ----------
    test_dataset : h5py.Dataset
        HDF5 dataset to be validated.
    golden_dataset : h5py.Dataset
        HDF5 dataset to use as reference.
    pixels_failed_threshold : float, optional
        The threshold of the percentage of pixels that can fail the comparison. Defaults
        to 0.01.
    diff_threshold : float, optional
        The abs. difference threshold between pixels to consider failing. Defaults to
        1e-5.

    Raises
    ------
    ComparisonError
        If the two datasets do not match.

    """
    golden = golden_dataset[()]
    test = test_dataset[()]
    if golden.dtype.kind == "S":
        if not np.array_equal(golden, test):
            raise ComparisonError(f"Dataset {golden_dataset.name} values do not match")
        return

    img_gold = np.ma.masked_invalid(golden)
    img_test = np.ma.masked_invalid(test)
    abs_diff = np.abs((img_gold.filled(0) - img_test.filled(0)))
    num_failed = np.count_nonzero(abs_diff > diff_threshold)
    # num_pixels = np.count_nonzero(~np.isnan(img_gold))  # do i want this?
    num_pixels = img_gold.size
    if num_failed / num_pixels > pixels_failed_threshold:
        raise ComparisonError(
            f"Dataset {golden_dataset.name} values do not match: Number of"
            f" pixels failed: {num_failed} / {num_pixels} ="
            f" {100*num_failed / num_pixels:.2f}%"
        )


def _check_raster_geometadata(golden_file: Filename, test_file: Filename) -> None:
    """Check if the raster metadata (bounds, CRS, and GT) match.

    Parameters
    ----------
    golden_file : Filename
        Path to the golden file.
    test_file : Filename
        Path to the test file to be compared.

    Raises
    ------
    ComparisonError
        If the two files do not match in their metadata

    """
    funcs = [io.get_raster_bounds, io.get_raster_crs, io.get_raster_gt]
    for func in funcs:
        val_golden = func(golden_file)  # type: ignore
        val_test = func(test_file)  # type: ignore
        if val_golden != val_test:
            raise ComparisonError(f"{func} does not match: {val_golden} vs {val_test}")


def _check_compressed_slc_dirs(golden: Filename, test: Filename) -> None:
    """Check if the compressed SLC directories match.

    Assumes that the compressed SLC directories are in the same directory as the
    `golden` and `test` product files, with the directory name `compressed_slcs`.

    Parameters
    ----------
    golden : Filename
        Path to the golden file.
    test : Filename
        Path to the test file to be compared.

    Raises
    ------
    ComparisonError
        If file names do not match in their compressed SLC directories

    """
    golden_slc_dir = Path(golden).parent / "compressed_slcs"
    test_slc_dir = Path(test).parent / "compressed_slcs"

    if not golden_slc_dir.exists():
        logger.info("No compressed SLC directory found in golden product.")
        return
    if not test_slc_dir.exists():
        raise ComparisonError(
            f"{test_slc_dir} does not exist, but {golden_slc_dir} exists."
        )

    golden_slc_names = [p.name for p in golden_slc_dir.iterdir()]
    test_slc_names = [p.name for p in test_slc_dir.iterdir()]

    if set(golden_slc_names) != set(test_slc_names):
        raise ComparisonError(
            f"Compressed SLC directories do not match: {golden_slc_names} vs"
            f" {test_slc_names}"
        )


def compare(golden: Filename, test: Filename, data_dset: str = DSET_DEFAULT) -> None:
    """Compare two HDF5 files for consistency."""
    logger.info("Comparing HDF5 contents...")
    with h5py.File(golden, "r") as hf_g, h5py.File(test, "r") as hf_t:
        compare_groups(hf_g, hf_t)

    logger.info("Checking geospatial metadata...")
    _check_raster_geometadata(
        io.format_nc_filename(golden, data_dset),
        io.format_nc_filename(test, data_dset),
    )

    logger.info(f"Files {golden} and {test} match.")
    _check_compressed_slc_dirs(golden, test)


def _validate_against_igram(
    product_file: Filename,
    igram_file: Filename,
    *,
    atol: float = 1e-6,
) -> None:
    """Check that the unwrapped phase is congruent with the specified interferogram.

    Parameters
    ----------
    product_file : Filename
        The file path of the product to validate.
    igram_file : Filename
        The file path of an interferogram dataset that the unwrapped phase must be
        congruent with.
    atol : float, optional
        Maximum allowable absolute error between the unwrapped and reference phase
        values (after re-wrapping), in radians. Must be nonnegative. Defaults to 1e-6.

    """
    logger.info("Checking for congruence with wrapped phase...")

    # Get unwrapped phase and connected component label data from the product.
    with h5py.File(product_file, mode="r") as f:
        unw = f["unwrapped_phase"][()]
        conncomp = f["connected_component_labels"][()]

    # Get a mask of valid pixels (that had nonzero CC label) and NaN-valued pixels.
    valid_mask = conncomp != 0
    nan_mask = np.isnan(unw)

    # Get wrapped phase data, in radians.
    logger.info(f"Interferogram file: {igram_file}")
    igram = io.load_gdal(igram_file)
    wrapped = np.angle(igram)

    # Check that the unwrapped phase is congruent with the wrapped phase.
    _check_phase_congruence(
        unw=unw,
        ref=wrapped,
        mask=(valid_mask & ~nan_mask),
        atol=atol,
    )


def _get_frame_id(hdf5_file: Filename) -> int:
    """Get the frame ID of an OPERA DISP-S1 product.

    Parameters
    ----------
    hdf5_file : Filename
        The product file path.

    Returns
    -------
    int
        The frame ID number of the product.

    """
    with h5py.File(hdf5_file, mode="r") as f:
        return f["identification/frame_id"][()]


def _check_frame_bounds(
    filename: Filename,
    frame_id: int,
    json_file: Filename | None = None,
    *,
    atol: float = 1e-6,
) -> None:
    """Validate a dataset's spatial reference and bounding box.

    Compare the EPSG code and spatial extents of the product with the expected bounds
    from the frame-to-burst JSON file.

    Parameters
    ----------
    filename : Filename
        The file path of the raster dataset to validate.
    frame_id : int
        The frame ID number of the dataset.
    json_file : Filename or None, optional
        The file path of the frame-to-burst JSON file. If None, uses the vendored
        frame-to-burst file. Defaults to None.
    atol : float, optional
        The absolute tolerance used for comparing bounding box coordinates. Must be
        nonnegative. Defaults to 1e-6.

    Raises
    ------
    ValidationError
        If the EPSG code of the raster dataset did not match the EPSG code of the
        corresponding frame in the JSON file.
    ValidationError
        If the bounding box of the raster dataset did not match the bounding box of the
        corresponding frame in the JSON file.

    """
    logger.info("Checking frame bounds again JSON file...")

    if atol < 0.0:
        errmsg = f"atol must be >= 0, got {atol}"
        raise ValueError(errmsg)

    if json_file is None:
        json_file = FRAME_TO_BURST_JSON_FILE

    logger.info(f"Dataset name: {filename}")
    logger.info(f"Frame ID: {frame_id}")
    logger.info(f"Frame-to-burst JSON file: {json_file}")

    # Extract the EPSG code and bounding box for this frame from the JSON file.
    json_epsg, json_bbox = get_frame_bbox(frame_id=frame_id, json_file=json_file)

    # Check EPSG code.
    data_epsg = io.get_raster_crs(filename=filename).to_epsg()
    logger.info(f"Product EPSG code: {data_epsg}")
    logger.info(f"Expected frame EPSG code: {json_epsg}")
    if data_epsg != json_epsg:
        errmsg = (
            f"product EPSG code ({data_epsg}) did not match expected EPSG code"
            f" {json_epsg}"
        )
        raise ValidationError(errmsg)

    # Check bounding box.
    data_bbox = io.get_raster_bounds(filename=filename)
    logger.info(f"Product bounds: {data_bbox}")
    logger.info(f"Expected frame bounds: {json_bbox}")
    if not np.allclose(data_bbox, json_bbox, rtol=0.0, atol=atol):
        errmsg = (
            "product bounding box did not match expected bounding box with absolute"
            f" tolerance {atol}\nproduct bbox: {data_bbox}\nexpected bbox: {json_bbox}"
        )
        raise ValidationError(errmsg)


def validate(
    product_file: Filename,
    golden_file: Filename | None = None,
    igram_file: Filename | None = None,
    json_file: Filename | None = None,
    data_dset: str = DSET_DEFAULT,
) -> None:
    r"""Validate an OPERA DISP-S1 product.

    The following validation checks are performed:

    1. Compares the product contents with a "golden" reference dataset, if one is
       provided.
    2. Optionally checks that the unwrapped phase is congruent with a supplied
       interferogram (i.e. their phase values differ only by integer multiples of
       :math:`2\pi`).
    3. Checks the frame spatial reference and bounds against the frame-to-burst JSON
       file.

    Parameters
    ----------
    product_file : Filename
        The file path of the product to validate.
    golden_file : Filename or None, optional
        The file path of a reference product to compare against. If none is specified,
        this check is skipped. Defaults to None.
    igram_file : Filename or None, optional
        The file path of an interferogram dataset that the unwrapped phase must be
        congruent with. If None, this check is skipped. Defaults to None.
    json_file : Filename or None, optional
        The file path of the frame-to-burst JSON file. If None, uses the vendored
        frame-to-burst file. Defaults to None.
    data_dset : str, optional
        The name of a particular dataset within the product to validate. Defaults to the
        unwrapped phase dataset.

    """
    # Compare product against golden reference.
    if golden_file is not None:
        compare(golden=golden_file, test=product_file, data_dset=data_dset)

    # Check the unwrapped phase data for congruence with the specified interferogram.
    # XXX Use a high tolerance here due to a known issue with SNAPHU that may cause
    # relatively large numerical errors (on the order of ~1e-3 radians) in the unwrapped
    # phase.
    if igram_file is not None:
        _validate_against_igram(
            product_file=product_file,
            igram_file=igram_file,
            atol=0.01,
        )

    # Check spatial reference & bounding box against frame-to-burst JSON file.
    frame_id = _get_frame_id(product_file)
    _check_frame_bounds(
        filename=io.format_nc_filename(product_file, data_dset),
        frame_id=frame_id,
        json_file=json_file,
    )
