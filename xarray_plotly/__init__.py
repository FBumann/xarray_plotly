"""
xarray-plotly: Interactive Plotly Express plotting accessor for xarray.

This package provides a `pxplot` accessor for xarray DataArray objects that
enables interactive plotting using Plotly Express.

Examples
--------
>>> import xarray as xr
>>> import numpy as np
>>> import xarray_plotly  # registers the accessor

>>> da = xr.DataArray(
...     np.random.rand(10, 3, 2),
...     dims=["time", "city", "scenario"],
... )

>>> # Auto-assignment: time->x, city->color, scenario->facet_col
>>> fig = da.pxplot.line()

>>> # Explicit assignment
>>> fig = da.pxplot.line(x="time", color="scenario", facet_col="city")

>>> # Skip a slot
>>> fig = da.pxplot.line(color=None)  # time->x, city->facet_col, scenario->facet_row
"""

from xarray import register_dataarray_accessor

from xarray_plotly.accessor import DataArrayPlotlyAccessor
from xarray_plotly.common import SLOT_ORDERS, auto

__all__ = [
    "SLOT_ORDERS",
    "DataArrayPlotlyAccessor",
    "auto",
]

__version__ = "0.1.0"

# Register the accessor
register_dataarray_accessor("pxplot")(DataArrayPlotlyAccessor)
