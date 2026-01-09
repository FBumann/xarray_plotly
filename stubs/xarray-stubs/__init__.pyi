"""Type stubs that add the plotly accessor to xarray.DataArray."""

from typing import Any

import plotly.graph_objects as go
from xarray import DataArray as _DataArray

class _SlotValue:
    """Type for slot values (auto, str, or None)."""

    ...

class DataArrayPlotlyAccessor:
    """Plotly Express plotting accessor for DataArray."""

    def __init__(self, darray: _DataArray) -> None: ...
    def line(
        self,
        *,
        x: _SlotValue | str | None = ...,
        color: _SlotValue | str | None = ...,
        line_dash: _SlotValue | str | None = ...,
        symbol: _SlotValue | str | None = ...,
        facet_col: _SlotValue | str | None = ...,
        facet_row: _SlotValue | str | None = ...,
        animation_frame: _SlotValue | str | None = ...,
        **px_kwargs: Any,
    ) -> go.Figure: ...
    def bar(
        self,
        *,
        x: _SlotValue | str | None = ...,
        color: _SlotValue | str | None = ...,
        pattern_shape: _SlotValue | str | None = ...,
        facet_col: _SlotValue | str | None = ...,
        facet_row: _SlotValue | str | None = ...,
        animation_frame: _SlotValue | str | None = ...,
        **px_kwargs: Any,
    ) -> go.Figure: ...
    def area(
        self,
        *,
        x: _SlotValue | str | None = ...,
        color: _SlotValue | str | None = ...,
        pattern_shape: _SlotValue | str | None = ...,
        facet_col: _SlotValue | str | None = ...,
        facet_row: _SlotValue | str | None = ...,
        animation_frame: _SlotValue | str | None = ...,
        **px_kwargs: Any,
    ) -> go.Figure: ...
    def scatter(
        self,
        *,
        x: _SlotValue | str | None = ...,
        y: _SlotValue | str | None = ...,
        color: _SlotValue | str | None = ...,
        size: _SlotValue | str | None = ...,
        symbol: _SlotValue | str | None = ...,
        facet_col: _SlotValue | str | None = ...,
        facet_row: _SlotValue | str | None = ...,
        animation_frame: _SlotValue | str | None = ...,
        **px_kwargs: Any,
    ) -> go.Figure: ...
    def box(
        self,
        *,
        x: _SlotValue | str | None = ...,
        color: _SlotValue | str | None = ...,
        facet_col: _SlotValue | str | None = ...,
        facet_row: _SlotValue | str | None = ...,
        animation_frame: _SlotValue | str | None = ...,
        **px_kwargs: Any,
    ) -> go.Figure: ...
    def imshow(
        self,
        *,
        x: _SlotValue | str | None = ...,
        y: _SlotValue | str | None = ...,
        facet_col: _SlotValue | str | None = ...,
        animation_frame: _SlotValue | str | None = ...,
        **px_kwargs: Any,
    ) -> go.Figure: ...

class DataArray(_DataArray):
    """DataArray with plotly accessor."""

    @property
    def plotly(self) -> DataArrayPlotlyAccessor: ...
