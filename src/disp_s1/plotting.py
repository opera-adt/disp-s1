from io import BytesIO
from typing import Any

import h5py
import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import rioxarray
from dolphin._types import Filename


def plot_product(
    filename: Filename,
    downsample=3,
    mask_on_conncomp: bool = True,
    figsize: tuple[float, float] = (9, 6),
    unwrapped_phase_cmap: str = "RdBu",
    unwrapped_phase_limits: tuple[float, float] = (-20, 20),
):
    """Plot the raster layers from one DISP product.

    Parameters
    ----------
    filename : Filename
        Path to the DISP product.
    downsample : int, optional
        Downsample factor, by default 3.
    mask_on_conncomp : bool, optional
        Mask the data where connected component label = 0, by default True.
        Otherwise, Mask the data where the data value = 0/nan.
    figsize : tuple[float, float], optional
        Figure size, by default (9, 6).
    unwrapped_phase_cmap : str, optional
        Colormap for the unwrapped phase, by default "RdBu".
    unwrapped_phase_limits : tuple[float, float], optional
        Limits for the unwrapped phase colormap, by default (-20, 20).

    Returns
    -------
    fig, axes : matplotlib.pyplot.Figure, matplotlib.pyplot.Axes
        Figure and axes objects.

    """
    ds = rioxarray.open_rasterio(filename, masked=True).sel(band=1)

    dsets = [
        "unwrapped_phase",
        "connected_component_labels",
        "temporal_coherence",
        "interferometric_correlation",
        "persistent_scatterer_mask",
    ]
    cmaps = [unwrapped_phase_cmap, "tab10", "viridis", "plasma", "gray"]

    vms = [unwrapped_phase_limits, (0, None), (0, 1), (0, 1), (0, 1)]

    if mask_on_conncomp:
        bad_mask = ds["connected_component_labels"][::downsample, ::downsample] == 0

    fig, axes = plt.subplots(
        ncols=2, nrows=3, sharex=True, sharey=True, figsize=figsize
    )

    for ax, dset_name, cmap, vm in zip(axes.ravel(), dsets, cmaps, vms):
        dset = ds[dset_name][::downsample, ::downsample]
        if not mask_on_conncomp:
            bad_mask = dset == 0

        dset.where(~bad_mask, np.nan).plot.imshow(
            ax=ax,
            cmap=cmap,
            vmin=vm[0],
            vmax=vm[1],
            cbar_kwargs={"label": dset.attrs["units"]},
        )
        ax.set_title(dset.attrs["long_name"])
        ax.set_aspect("equal")

    fig.tight_layout()
    return fig, axes


class HDF5Explorer:
    """Class which maps an HDF5 file and allows tab-completion to explore datasets."""

    def __init__(self, hdf5_filepath: str, load_less_than: float = 1e3):  # noqa: D107
        self.hdf5_filepath = hdf5_filepath
        self._hf = h5py.File(hdf5_filepath, "r")
        self._root_group = _HDF5GroupExplorer(
            self._hf["/"], load_less_than=load_less_than
        )

    def close(self):  # noqa: D102
        self._hf.close()

    def __getattr__(self, name):
        return getattr(self._root_group, name)

    def __dir__(self):
        return self._root_group.__dir__()

    def __repr__(self):
        return f"HDF5Explorer({self.hdf5_filepath})"


class _HDF5GroupExplorer:
    """Internal class to explore a group in an HDF5 file."""

    def __init__(self, group: h5py.Group, load_less_than: float = 1e3):
        self._group = group
        self._attr_cache: dict[str, Any] = {}
        self._populate_attr_cache(load_less_than)

    @property
    def group_path(self) -> str:
        return self._group.name

    def _populate_attr_cache(self, load_less_than: float = 1e3):
        for name, item in self._group.items():
            if isinstance(item, h5py.Group):
                self._attr_cache[name] = _HDF5GroupExplorer(item)
            elif isinstance(item, h5py.Dataset):
                if item.size < load_less_than:
                    self._attr_cache[name] = item[()]
                else:
                    self._attr_cache[name] = item
            else:
                self._attr_cache[name] = item

    def __getattr__(self, name):
        if name not in self._attr_cache:
            raise AttributeError(
                f"'{name}' not found in the group '{self.group_path}'."
            )
        return self._attr_cache[name]

    def __dir__(self):
        return list(self._attr_cache.keys())


def create_explorer_widget(
    hf: h5py.File, load_less_than: float = 1e3
) -> widgets.Widget:
    """Make a widget in Jupyter to explore a h5py file.

    Examples
    --------
    >>> hf = h5py.File("file.h5", "r")
    >>> create_explorer_widget(hf)

    """

    def _make_thumbnail(image) -> widgets.Image:
        # Create a thumbnail of the dataset
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(image, cmap="gray", vmax=np.nanpercentile(image, 99))
        ax.axis("off")
        buf = BytesIO()
        plt.savefig(buf, format="png", dpi=150)
        plt.close(fig)
        buf.seek(0)
        # Display the thumbnail in an Image widget
        return widgets.Image(value=buf.read(), format="png")

    def _add_widgets(item, level: int = 0) -> widgets.Widget:
        """Recursively add widgets to the accordion widget."""
        if isinstance(item, h5py.Group):
            # Add a new accordion widget for the group
            accordion = widgets.Accordion(selected_index=None)
            for key, value in item.items():
                widget = _add_widgets(value, level + 1)
                accordion.children += (widget,)
                accordion.set_title(len(accordion.children) - 1, key)
            return accordion

        # Once we're at a leaf node, add a widget for the dataset
        elif isinstance(item, h5py.Dataset):
            attributes = [f"<b>{k}:</b> {v}" for k, v in item.attrs.items()]
            content = f"Type: {item.dtype}<br>Shape: {item.shape}<br>"
            content += "<br>".join(attributes)
            if item.size < load_less_than:
                content += f"<br>Value: {item[()]}"
            html_widget = widgets.HTML(content)

            if not item.ndim == 2 or not item.dtype == np.complex64:
                return html_widget
            # If the dataset is a 2D complex array, make a thumbnail
            image_widget = _make_thumbnail(np.abs(item[::5, ::10]))
            return widgets.VBox([image_widget, html_widget])

        else:
            # Other types of items
            return widgets.HTML(f"{item}")

    # Now add everything starting at the root
    return _add_widgets(hf, 0)
