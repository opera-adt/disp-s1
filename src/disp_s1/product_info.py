from dataclasses import dataclass, fields

import numpy as np
from numpy.typing import DTypeLike


@dataclass(frozen=True)
class DispProductInfo:
    """Describe me plz."""

    name: str
    description: str
    fillvalue: DTypeLike
    attrs: dict

    @classmethod
    def unwrapped_phase(cls):
        return cls(
            name="unwrapped_phase",
            description="Unwrapped phase",
            fillvalue=np.nan,
            attrs=dict(units="radians"),
        )

    @classmethod
    def connected_component_labels(cls):
        return cls(
            name="connected_component_labels",
            description="Connected component labels of the unwrapped phase",
            fillvalue=0,
            attrs=dict(units="unitless"),
        )

    @classmethod
    def temporal_coherence(cls):
        return cls(
            name="temporal_coherence",
            description="Temporal coherence of phase inversion",
            fillvalue=np.nan,
            attrs=dict(units="unitless"),
        )

    @classmethod
    def interferometric_correlation(cls):
        return cls(
            name="interferometric_correlation",
            description=(
                "Estimate of interferometric correlation derived from"
                " multilooked interferogram."
            ),
            fillvalue=np.nan,
            attrs=dict(units="unitless"),
        )

    @classmethod
    def persistent_scattere_mask(cls):
        return cls(
            name="persistent_scatterer_mask",
            description=(
                "Mask of persistent scatterers downsampled to the multilooked"
                " output grid."
            ),
            fillvalue=255,
            attrs=dict(units="unitless"),
        )


@dataclass(frozen=True)
class DispProductsInfo:
    """Describe me plz."""

    unwrapped_phase: DispProductInfo = DispProductInfo.unwrapped_phase()
    connected_component_labels: DispProductInfo = (
        DispProductInfo.connected_component_labels()
    )
    temporal_coherence: DispProductInfo = DispProductInfo.temporal_coherence()
    interferometric_correlation: DispProductInfo = (
        DispProductInfo.interferometric_correlation()
    )
    persistent_scattere_mask: DispProductInfo = (
        DispProductInfo.persistent_scattere_mask()
    )

    def as_list(self):
        return [getattr(self, field) for field in fields(self)]
