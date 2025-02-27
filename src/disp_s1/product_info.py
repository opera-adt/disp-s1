from dataclasses import dataclass, field

import numpy as np
from dolphin.unwrap import DEFAULT_CCL_NODATA
from numpy.typing import DTypeLike


@dataclass
class ProductInfo:
    """Information about a displacement product dataset."""

    name: str
    long_name: str
    description: str
    fillvalue: DTypeLike
    dtype: DTypeLike
    attrs: dict[str, str] = field(default_factory=dict)
    keep_bits: int | None = None


@dataclass
class DisplacementProducts:
    """Container for displacement product dataset info."""

    displacement: ProductInfo = field(
        default_factory=lambda: ProductInfo(
            name="displacement",
            long_name="Line-of-sight displacement",
            description=(
                "Displacement along the radar Line-of-Sight (LOS) direction."
                " Positive values indicate apparent motion towards the platform."
            ),
            fillvalue=np.nan,
            attrs={"units": "meters"},
            # 12 bits, for random values in meters from -1 to 1, has a max
            # quantization error of about 0.06 millimeters
            keep_bits=12,
            dtype=np.float32,
        )
    )

    short_wavelength_displacement: ProductInfo = field(
        default_factory=lambda: ProductInfo(
            name="short_wavelength_displacement",
            long_name="Short wavelength displacement",
            description=(
                "Displacement in Line-of-Sight (LOS) with long-wavelength signals"
                " removed. Positive values indicate apparent motion towards the"
                " platform."
            ),
            fillvalue=np.nan,
            attrs={"units": "meters"},
            # 12 bits, for random values in meters from -1 to 1, has a max
            # quantization error of about 0.06 millimeters
            keep_bits=12,
            dtype=np.float32,
        )
    )
    recommended_mask: ProductInfo = field(
        default_factory=lambda: ProductInfo(
            name="recommended_mask",
            long_name="Recommended Mask",
            description=(
                "Suggested mask to remove low quality pixels, where 0 indicates a bad"
                " pixel, 1 is a good pixel. Pixels are set to 0 for three reasons: 1)"
                " the pixel is marked as water in /water_mask, 2) the pixel is 0 in"
                " /connected_component_labels, or 3) the pixel has low"
                " /temporal_coherence and low /phase_similarity, where 'low' is below"
                " the threshold listed in this dataset's attributes."
            ),
            fillvalue=255,
            attrs={"units": "unitless"},
            dtype=np.uint8,
        )
    )
    connected_component_labels: ProductInfo = field(
        default_factory=lambda: ProductInfo(
            name="connected_component_labels",
            long_name="Connected Component Labels",
            description="Connected component labels of the unwrapped phase",
            fillvalue=DEFAULT_CCL_NODATA,
            attrs={"units": "unitless"},
            dtype=np.uint16,
        )
    )
    temporal_coherence: ProductInfo = field(
        default_factory=lambda: ProductInfo(
            name="temporal_coherence",
            long_name="Temporal Coherence",
            description="Temporal coherence of phase inversion",
            fillvalue=np.nan,
            attrs={"units": "unitless"},
            # 8 bits (between 0 and 1) is around .001 precision
            keep_bits=8,
            dtype=np.float32,
        )
    )
    interferometric_correlation: ProductInfo = field(
        default_factory=lambda: ProductInfo(
            name="estimated_phase_quality",
            long_name="Estimated phase quality",
            description="Sliding window estimate of multi-looked phase noise",
            fillvalue=np.nan,
            attrs={"units": "unitless"},
            # 8 bits (between 0 and 1) is around .001 precision
            keep_bits=8,
            dtype=np.float32,
        )
    )
    persistent_scatterer_mask: ProductInfo = field(
        default_factory=lambda: ProductInfo(
            name="persistent_scatterer_mask",
            long_name="Persistent Scatterer Mask",
            description=(
                "Mask of persistent scatterers downsampled to the multilooked output"
                " grid"
            ),
            fillvalue=255,
            attrs={"units": "unitless"},
            dtype=np.uint8,
        )
    )
    shp_counts: ProductInfo = field(
        default_factory=lambda: ProductInfo(
            name="shp_counts",
            long_name="Statistically Homogeneous Pixels Counts",
            description=(
                "Number of statistically homogeneous pixels (SHPs) used at each output"
                " pixel during multilooking"
            ),
            fillvalue=0,
            attrs={"units": "unitless"},
            dtype=np.int16,
        )
    )
    water_mask: ProductInfo = field(
        default_factory=lambda: ProductInfo(
            name="water_mask",
            long_name="Water Mask",
            description="Binary mask created to ignore water during processing.",
            fillvalue=255,
            attrs={"units": "unitless"},
            dtype=np.uint8,
        )
    )
    phase_similarity: ProductInfo = field(
        default_factory=lambda: ProductInfo(
            name="phase_similarity",
            long_name="Phase Similarity",
            description=(
                "Median cosine similarity of wrapped phase in each pixel's"
                " neighborhood."
            ),
            fillvalue=np.nan,
            attrs={"units": "unitless"},
            # 8 bits (between 0 and 1) is around .001 precision
            keep_bits=8,
            dtype=np.float32,
        )
    )
    timeseries_inversion_residuals: ProductInfo = field(
        default_factory=lambda: ProductInfo(
            name="timeseries_inversion_residuals",
            long_name="Timeseries inversion residuals",
            description=(
                "Sum of residuals of the timeseries inversion for this secondary date."
            ),
            fillvalue=np.nan,
            attrs={"units": "radians"},
            # 8 bits (between 0 and 1) is around .001 precision
            keep_bits=7,
            dtype=np.float32,
        )
    )

    def __iter__(self):
        """Return all displacement dataset info as an iterable."""
        return iter(self.__dict__.values())

    @property
    def names(self) -> list[str]:
        """Return all displacement dataset names as a list."""
        return [v.name for v in self.__dict__.values()]


# Create a single instance to be used throughout the application
DISPLACEMENT_PRODUCTS = DisplacementProducts()
