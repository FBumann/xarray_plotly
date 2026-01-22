"""Interactive Plotly Express plotting for xarray.

This package provides a `plotly` accessor for xarray DataArray and Dataset objects,
enabling interactive visualization with Plotly Express.

Features:
    - **Interactive plots**: Zoom, pan, hover, toggle traces
    - **Automatic dimension assignment**: Dimensions fill slots (x, color, facet) by position
    - **Multiple plot types**: line, bar, area, scatter, box, imshow
    - **Dataset support**: Plot all variables at once with "variable" dimension
    - **Faceting and animation**: Built-in subplot grids and animated plots
    - **Customizable**: Returns Plotly Figure objects for further modification

Usage:
    Accessor style::

        import xarray_plotly
        fig = da.plotly.line()
        fig = ds.plotly.line()  # Dataset: all variables

    Function style (recommended for IDE completion)::

        from xarray_plotly import xpx
        fig = xpx(da).line()
        fig = xpx(ds).line()  # Dataset: all variables

Example:
    ```python
    import xarray as xr
    import numpy as np
    from xarray_plotly import xpx

    da = xr.DataArray(
        np.random.rand(10, 3, 2),
        dims=["time", "city", "scenario"],
    )
    fig = xpx(da).line()  # Auto: time->x, city->color, scenario->facet_col
    fig = xpx(da).line(x="time", color="scenario")  # Explicit
    fig = xpx(da).line(color=None)  # Skip slot

    # Dataset: plot all variables (accessor or xpx)
    ds = xr.Dataset({"temp": da, "precip": da})
    fig = xpx(ds).line()  # "variable" dimension for color
    fig = xpx(ds).line(facet_col="variable")  # Facet by variable
    ```
"""

from importlib.metadata import version
from typing import overload

from xarray import DataArray, Dataset, register_dataarray_accessor, register_dataset_accessor

from xarray_plotly import config
from xarray_plotly.accessor import DataArrayPlotlyAccessor, DatasetPlotlyAccessor
from xarray_plotly.common import SLOT_ORDERS, auto
from xarray_plotly.figures import (
    add_secondary_y,
    overlay,
    update_animation_traces,
)

__all__ = [
    "SLOT_ORDERS",
    "add_secondary_y",
    "auto",
    "config",
    "overlay",
    "update_animation_traces",
    "xpx",
]


@overload
def xpx(data: DataArray) -> DataArrayPlotlyAccessor: ...


@overload
def xpx(data: Dataset) -> DatasetPlotlyAccessor: ...


def xpx(data: DataArray | Dataset) -> DataArrayPlotlyAccessor | DatasetPlotlyAccessor:
    """Get the plotly accessor for a DataArray or Dataset with full IDE code completion.

    This is an alternative to `da.plotly` / `ds.plotly` that provides proper type hints
    and code completion in IDEs.

    Args:
        data: The DataArray or Dataset to plot.

    Returns:
        The accessor with plotting methods (line, bar, area, scatter, box, imshow).

    Example:
        ```python
        from xarray_plotly import xpx

        # DataArray
        fig = xpx(da).line()  # Full code completion works here

        # Dataset
        fig = xpx(ds).line()  # Plots all variables
        fig = xpx(ds).line(var="temperature")  # Single variable
        ```
    """
    if isinstance(data, Dataset):
        return DatasetPlotlyAccessor(data)
    return DataArrayPlotlyAccessor(data)


__version__ = version("xarray_plotly")

# Register the accessors
register_dataarray_accessor("plotly")(DataArrayPlotlyAccessor)
register_dataset_accessor("plotly")(DatasetPlotlyAccessor)
