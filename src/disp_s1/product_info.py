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
    attrs: dict[str, str] = field(default_factory=dict)


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
        )
    )

    connected_component_labels: ProductInfo = field(
        default_factory=lambda: ProductInfo(
            name="connected_component_labels",
            description="Connected component labels of the unwrapped phase",
            fillvalue=DEFAULT_CCL_NODATA,
            attrs={"units": "unitless"},
        )
    )

    temporal_coherence: ProductInfo = field(
        default_factory=lambda: ProductInfo(
            name="temporal_coherence",
            description="Temporal coherence of phase inversion",
            fillvalue=np.nan,
            attrs={"units": "unitless"},
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
