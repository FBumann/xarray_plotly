"""Partial type stubs for xarray - adds plotly accessor to DataArray."""

from typing import Any

from xarray_plotly.accessor import DataArrayPlotlyAccessor

# Re-export everything from xarray
from xarray import *

# Augment DataArray with the plotly accessor
class DataArray(DataArray):
    @property
    def plotly(self) -> DataArrayPlotlyAccessor: ...
