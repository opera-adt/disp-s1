from io import BytesIO
from typing import Any

import h5py
import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import rioxarray
from dolphin._types import Filename

from disp_s1.browse_image import DEFAULT_CMAP


def plot_product(
    filename: Filename,
    downsample=3,
    use_recommended_mask: bool = True,
    figsize: tuple[float, float] = (10, 10),
    disp_cmap: str = DEFAULT_CMAP,
    disp_limits: tuple[float, float] = (-0.20, 0.20),
    filtered_disp_limits: tuple[float, float] = (-0.05, 0.05),
):
    """Plot the raster layers from one DISP product.

    Parameters
    ----------
    filename : Filename
        Path to the DISP product.
    downsample : int, optional
        Downsample factor, by default 3.
    use_recommended_mask : bool, optional
        Mask the data using the `recommended_mask` layer.
        Otherwise, Mask the data where the data value = 0/nan.
    figsize : tuple[float, float], optional
        Figure size, by default (9, 6).
    disp_cmap : str, optional
        Colormap for the unwrapped phase, by default Vik
        https://cmap-docs.readthedocs.io/en/latest/catalog/diverging/crameri:vik/
    disp_limits : tuple[float, float], optional
        Limits (in meters) for the displacement phase colormap.
        Default is (-0.2, 0.2)
    filtered_disp_limits : tuple[float, float], optional
        Limits (in meters) for the short_wavelength_displacement phase colormap.
        Default is (-0.05, 0.05)

    Returns
    -------
    fig, axes : matplotlib.pyplot.Figure, matplotlib.pyplot.Axes
        Figure and axes objects.

    """
    # from disp_s1.product_info import DISPLACEMENT_PRODUCTS
    # product_infos = list(DISPLACEMENT_PRODUCTS)
    ds = rioxarray.open_rasterio(filename, masked=True).sel(band=1)

    dsets = [
        "displacement",
        "short_wavelength_displacement",
        "connected_component_labels",
        "temporal_coherence",
        "estimated_phase_quality",
        "phase_similarity",
        "persistent_scatterer_mask",
        "unwrapper_mask",
        "water_mask",
        "recommended_mask",
    ]
    cmaps = [
        disp_cmap,
        disp_cmap,
        "tab10",
        # Quality masks
        "plasma_r",
        "plasma_r",
        "plasma_r",
        # binary masks
        "gray_r",
        "viridis",
        "viridis",
        "viridis",
    ]

    vms = [
        disp_limits,
        filtered_disp_limits,
        (0, None),
        (0, 1),
        (0, 1),
        (0, 1),
        (0, 1),
        (0, 1),
        (0, 1),
        (0, 1),
    ]

    if use_recommended_mask:
        bad_mask = ds["recommended_mask"][::downsample, ::downsample] == 0

    fig, axes = plt.subplots(
        ncols=4, nrows=3, sharex=True, sharey=True, figsize=figsize
    )

    for ax, dset_name, cmap, vm in zip(
        axes.ravel()[: len(dsets)], dsets, cmaps, vms, strict=True
    ):
        dset = ds[dset_name][::downsample, ::downsample]
        if not use_recommended_mask:
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
    hf: h5py.File,
    load_less_than: float = 1e3,
    subsample_factor: tuple[int, int] = (20, 20),
    thumbnail_width="600px",
) -> widgets.Widget:
    """Make a widget in Jupyter to explore a h5py file.

    Examples
    --------
    >>> hf = h5py.File("file.h5", "r")
    >>> create_explorer_widget(hf)

    """

    def _make_thumbnail(image) -> widgets.Image:
        # Create a thumbnail of the dataset
        fig, ax = plt.subplots(figsize=(5, 4))
        vmax = np.nanpercentile(image, 99)
        vmin = np.nanpercentile(image, 1)
        ax.imshow(image, cmap="gray", vmax=vmax, vmin=vmin)
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

            if not item.ndim == 2:
                return html_widget
            # make a thumbnail
            # If the dataset is a 2D complex array, use abs
            sub_r, sub_c = subsample_factor
            arr = item[::sub_r, ::sub_c]
            if item.dtype == np.complex64:
                image_widget = _make_thumbnail(np.abs(arr))
            else:
                image_widget = _make_thumbnail(arr)
            return widgets.VBox(
                [image_widget, html_widget],
                layout=widgets.Layout(width=thumbnail_width),
            )

        else:
            # Other types of items
            return widgets.HTML(f"{item}")

    # Now add everything starting at the root
    return _add_widgets(hf, 0)
