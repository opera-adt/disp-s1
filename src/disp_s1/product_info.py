from dataclasses import dataclass, field

import numpy as np
from dolphin.unwrap import DEFAULT_CCL_NODATA
from numpy.typing import DTypeLike


@dataclass
class ProductInfo:
    """Information about a displacement product dataset."""

    name: str
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
            description=(
                "Displacement with noise in Line-of-Sight (LOS)."
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
    connected_component_labels: ProductInfo = field(
        default_factory=lambda: ProductInfo(
            name="connected_component_labels",
            description="Connected component labels of the unwrapped phase",
            fillvalue=DEFAULT_CCL_NODATA,
            attrs={"units": "unitless"},
            dtype=np.uint16,
        )
    )
    temporal_coherence: ProductInfo = field(
        default_factory=lambda: ProductInfo(
            name="temporal_coherence",
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
            name="interferometric_correlation",
            description=(
                "Estimate of interferometric correlation derived from multilooked"
                " interferogram."
            ),
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
            description=(
                "Mask of persistent scatterers downsampled to the multilooked output"
                " grid."
            ),
            fillvalue=255,
            attrs={"units": "unitless"},
            dtype=np.uint8,
        )
    )
    shp_counts: ProductInfo = field(
        default_factory=lambda: ProductInfo(
            name="shp_counts",
            description=(
                "Number of statistically homogeneous pixels (SHPs) used during"
                " multilooking."
            ),
            fillvalue=0,
            attrs={"units": "unitless"},
            dtype=np.int32,
        )
    )
    unwrapper_mask: ProductInfo = field(
        default_factory=lambda: ProductInfo(
            name="unwrapper_mask",
            description="Mask used during phase unwrapping to ignore input pixels.",
            fillvalue=255,
            attrs={"units": "unitless"},
            dtype=np.uint8,
        )
    )

    def __iter__(self):
        """Return all displacement dataset info as an iterable."""
        return iter(self.__dict__.values())

    @property
    def names(self) -> list[str]:
        """Return all displacement dataset names as a list."""
        return list(self.__dict__.keys())


# Create a single instance to be used throughout the application
DISPLACEMENT_PRODUCTS = DisplacementProducts()
