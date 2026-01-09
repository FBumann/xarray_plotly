"""
xarray_plotly: Interactive Plotly Express plotting for xarray.

This package provides interactive plotting for xarray DataArray objects
using Plotly Express.

Examples
--------
>>> import xarray as xr
>>> import numpy as np
>>> from xarray_plotly import xpx

>>> da = xr.DataArray(
...     np.random.rand(10, 3, 2),
...     dims=["time", "city", "scenario"],
... )

>>> # Auto-assignment: time->x, city->color, scenario->facet_col
>>> fig = xpx(da).line()

>>> # Explicit assignment
>>> fig = xpx(da).line(x="time", color="scenario", facet_col="city")

>>> # Skip a slot
>>> fig = xpx(da).line(color=None)  # time->x, city->facet_col, scenario->facet_row
"""

from importlib.metadata import version

from xarray import DataArray, register_dataarray_accessor

from xarray_plotly import config
from xarray_plotly.accessor import DataArrayPlotlyAccessor
from xarray_plotly.common import SLOT_ORDERS, auto

__all__ = [
    "DataArrayPlotlyAccessor",
    "SLOT_ORDERS",
    "auto",
    "config",
    "xpx",
]


def xpx(da: DataArray) -> DataArrayPlotlyAccessor:
    """
    Get the plotly accessor for a DataArray with full IDE code completion.

    This is an alternative to `da.plotly` that provides proper type hints
    and code completion in IDEs.

    Parameters
    ----------
    da : DataArray
        The DataArray to plot.

    Returns
    -------
    DataArrayPlotlyAccessor
        The accessor with plotting methods.

    Examples
    --------
    >>> from xarray_plotly import xpx
    >>> fig = xpx(da).line()  # Full code completion works here
    """
    return DataArrayPlotlyAccessor(da)


__version__ = version("xarray_plotly")

# Register the accessor
register_dataarray_accessor("plotly")(DataArrayPlotlyAccessor)
