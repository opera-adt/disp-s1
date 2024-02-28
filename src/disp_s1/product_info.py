from dataclasses import dataclass, fields

import numpy as np
from dolphin.unwrap import DEFAULT_CCL_NODATA
from numpy.typing import DTypeLike


@dataclass(frozen=True)
class DispProductInfo:
    """Container for items used in creating displacement product datasets."""

    # Name of of the dataset.
    name: str
    # Description of the dataset.
    description: str
    # Fill value of the dataset.
    fillvalue: DTypeLike
    # Attributes of the dataset.
    attrs: dict

    @classmethod
    def unwrapped_phase(cls):
        """Return container of unwrapped phase specific information."""
        return cls(
            name="unwrapped_phase",
            description="Unwrapped phase",
            fillvalue=np.nan,
            attrs={"units": "radians"},
        )

    @classmethod
    def connected_component_labels(cls):
        """Return container of connected component label specific information."""
        return cls(
            name="connected_component_labels",
            description="Connected component labels of the unwrapped phase",
            fillvalue=DEFAULT_CCL_NODATA,
            attrs={"units": "unitless"},
        )

    @classmethod
    def temporal_coherence(cls):
        """Return container of temporal coherence specific information."""
        return cls(
            name="temporal_coherence",
            description="Temporal coherence of phase inversion",
            fillvalue=np.nan,
            attrs={"units": "unitless"},
        )

    @classmethod
    def interferometric_correlation(cls):
        """Return container of interferometric correlation specific information."""
        return cls(
            name="interferometric_correlation",
            description=(
                "Estimate of interferometric correlation derived from"
                " multilooked interferogram."
            ),
            fillvalue=np.nan,
            attrs={"units": "unitless"},
        )

    @classmethod
    def persistent_scatterer_mask(cls):
        """Return container of persistent scatterer mask specific information."""
        return cls(
            name="persistent_scatterer_mask",
            description=(
                "Mask of persistent scatterers downsampled to the multilooked"
                " output grid."
            ),
            fillvalue=255,
            attrs={"units": "unitless"},
        )


@dataclass(frozen=True)
class DispProductsInfo:
    """Container for instantiated displacement product dataset info containers."""

    unwrapped_phase: DispProductInfo = DispProductInfo.unwrapped_phase()
    connected_component_labels: (
        DispProductInfo
    ) = DispProductInfo.connected_component_labels()
    temporal_coherence: DispProductInfo = DispProductInfo.temporal_coherence()
    interferometric_correlation: (
        DispProductInfo
    ) = DispProductInfo.interferometric_correlation()
    persistent_scatterer_mask: (
        DispProductInfo
    ) = DispProductInfo.persistent_scatterer_mask()

    def as_list(self):
        """Return all displacement dataset info containers as a list."""
        return [getattr(self, field.name) for field in fields(self)]

    def product_names(self):
        """Return all displacement dataset names as a list."""
        return [field.name for field in fields(self)]


DISP_PRODUCTS_INFO = DispProductsInfo().as_list()
DISP_PRODUCT_NAMES = DispProductsInfo().product_names()
