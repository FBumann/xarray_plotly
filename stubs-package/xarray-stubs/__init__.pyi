"""Partial type stubs for xarray - adds plotly accessor to DataArray."""

# Re-export everything from xarray
from xarray import *  # noqa: F403

from xarray_plotly.accessor import DataArrayPlotlyAccessor

# Augment DataArray with the plotly accessor
class DataArray(DataArray):
    @property
    def plotly(self) -> DataArrayPlotlyAccessor: ...
