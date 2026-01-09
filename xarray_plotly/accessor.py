"""
Accessor classes for Plotly Express plotting on DataArray and Dataset.
"""

from typing import Any, ClassVar

import plotly.graph_objects as go
from xarray import DataArray

from xarray_plotly import plotting
from xarray_plotly.common import SlotValue, auto


class DataArrayPlotlyAccessor:
    """
    Enables use of Plotly Express plotting functions on a DataArray.

    Methods return Plotly Figure objects for interactive visualization.

    Examples
    --------
    >>> import xarray as xr
    >>> import numpy as np
    >>> da = xr.DataArray(np.random.rand(10, 3), dims=["time", "city"])
    >>> fig = da.plotly.line()  # Auto-assign dims: time->x, city->color
    >>> fig.show()

    >>> fig = da.plotly.line(x="time", color=None)  # Explicit assignment
    >>> fig.update_layout(title="My Plot")
    """

    __all__: ClassVar = ["line", "bar", "area", "scatter", "box", "imshow"]

    def __init__(self, darray: DataArray) -> None:
        self._da = darray

    def __dir__(self) -> list[str]:
        """List available plot methods."""
        return list(self.__all__) + list(super().__dir__())

    def line(
        self,
        *,
        x: SlotValue = auto,
        color: SlotValue = auto,
        line_dash: SlotValue = auto,
        symbol: SlotValue = auto,
        facet_col: SlotValue = auto,
        facet_row: SlotValue = auto,
        animation_frame: SlotValue = auto,
        **px_kwargs: Any,
    ) -> go.Figure:
        """
        Create an interactive line plot using Plotly Express.

        The y-axis always shows the DataArray values. Dimensions are assigned
        to other slots by their order:
        x -> color -> line_dash -> symbol -> facet_col -> facet_row -> animation_frame

        Parameters
        ----------
        x
            Dimension for x-axis. Default: first dimension.
        color
            Dimension for color grouping. Default: second dimension.
        line_dash
            Dimension for line dash style. Default: third dimension.
        symbol
            Dimension for marker symbol. Default: fourth dimension.
        facet_col
            Dimension for subplot columns. Default: fifth dimension.
        facet_row
            Dimension for subplot rows. Default: sixth dimension.
        animation_frame
            Dimension for animation. Default: seventh dimension.
        **px_kwargs
            Additional arguments passed to `plotly.express.line()`.

        Returns
        -------
        plotly.graph_objects.Figure
        """
        return plotting.line(
            self._da,
            x=x,
            color=color,
            line_dash=line_dash,
            symbol=symbol,
            facet_col=facet_col,
            facet_row=facet_row,
            animation_frame=animation_frame,
            **px_kwargs,
        )

    def bar(
        self,
        *,
        x: SlotValue = auto,
        color: SlotValue = auto,
        pattern_shape: SlotValue = auto,
        facet_col: SlotValue = auto,
        facet_row: SlotValue = auto,
        animation_frame: SlotValue = auto,
        **px_kwargs: Any,
    ) -> go.Figure:
        """
        Create an interactive bar chart using Plotly Express.

        The y-axis always shows the DataArray values. Dimensions are assigned
        to other slots by their order:
        x -> color -> pattern_shape -> facet_col -> facet_row -> animation_frame

        Parameters
        ----------
        x
            Dimension for x-axis. Default: first dimension.
        color
            Dimension for color grouping. Default: second dimension.
        pattern_shape
            Dimension for bar fill pattern. Default: third dimension.
        facet_col
            Dimension for subplot columns. Default: fourth dimension.
        facet_row
            Dimension for subplot rows. Default: fifth dimension.
        animation_frame
            Dimension for animation. Default: sixth dimension.
        **px_kwargs
            Additional arguments passed to `plotly.express.bar()`.

        Returns
        -------
        plotly.graph_objects.Figure
        """
        return plotting.bar(
            self._da,
            x=x,
            color=color,
            pattern_shape=pattern_shape,
            facet_col=facet_col,
            facet_row=facet_row,
            animation_frame=animation_frame,
            **px_kwargs,
        )

    def area(
        self,
        *,
        x: SlotValue = auto,
        color: SlotValue = auto,
        pattern_shape: SlotValue = auto,
        facet_col: SlotValue = auto,
        facet_row: SlotValue = auto,
        animation_frame: SlotValue = auto,
        **px_kwargs: Any,
    ) -> go.Figure:
        """
        Create an interactive stacked area chart using Plotly Express.

        The y-axis always shows the DataArray values. Dimensions are assigned
        to other slots by their order:
        x -> color -> pattern_shape -> facet_col -> facet_row -> animation_frame

        Parameters
        ----------
        x
            Dimension for x-axis. Default: first dimension.
        color
            Dimension for color/stacking. Default: second dimension.
        pattern_shape
            Dimension for fill pattern. Default: third dimension.
        facet_col
            Dimension for subplot columns. Default: fourth dimension.
        facet_row
            Dimension for subplot rows. Default: fifth dimension.
        animation_frame
            Dimension for animation. Default: sixth dimension.
        **px_kwargs
            Additional arguments passed to `plotly.express.area()`.

        Returns
        -------
        plotly.graph_objects.Figure
        """
        return plotting.area(
            self._da,
            x=x,
            color=color,
            pattern_shape=pattern_shape,
            facet_col=facet_col,
            facet_row=facet_row,
            animation_frame=animation_frame,
            **px_kwargs,
        )

    def scatter(
        self,
        *,
        x: SlotValue = auto,
        y: SlotValue | str = "value",
        color: SlotValue = auto,
        symbol: SlotValue = auto,
        facet_col: SlotValue = auto,
        facet_row: SlotValue = auto,
        animation_frame: SlotValue = auto,
        **px_kwargs: Any,
    ) -> go.Figure:
        """
        Create an interactive scatter plot using Plotly Express.

        By default, y-axis shows the DataArray values. Set y to a dimension
        name to create dimension-vs-dimension plots (e.g., lat vs lon colored
        by value).

        Parameters
        ----------
        x
            Dimension for x-axis. Default: first dimension.
        y
            What to plot on y-axis. Default "value" uses DataArray values.
            Can be a dimension name for dimension vs dimension plots.
        color
            Dimension for color grouping. Default: second dimension.
            Use "value" to color by DataArray values (useful with y=dimension).
        symbol
            Dimension for marker symbol. Default: third dimension.
        facet_col
            Dimension for subplot columns. Default: fourth dimension.
        facet_row
            Dimension for subplot rows. Default: fifth dimension.
        animation_frame
            Dimension for animation. Default: sixth dimension.
        **px_kwargs
            Additional arguments passed to `plotly.express.scatter()`.

        Returns
        -------
        plotly.graph_objects.Figure
        """
        return plotting.scatter(
            self._da,
            x=x,
            y=y,
            color=color,
            symbol=symbol,
            facet_col=facet_col,
            facet_row=facet_row,
            animation_frame=animation_frame,
            **px_kwargs,
        )

    def box(
        self,
        *,
        x: SlotValue = auto,
        color: SlotValue = None,
        facet_col: SlotValue = None,
        facet_row: SlotValue = None,
        animation_frame: SlotValue = None,
        **px_kwargs: Any,
    ) -> go.Figure:
        """
        Create an interactive box plot using Plotly Express.

        The y-axis always shows the DataArray values. By default, only the
        first dimension is assigned to x; all other dimensions are aggregated
        into the box statistics.

        Parameters
        ----------
        x
            Dimension for x-axis categories. Default: first dimension.
        color
            Dimension for color grouping. Default: None (aggregated).
        facet_col
            Dimension for subplot columns. Default: None (aggregated).
        facet_row
            Dimension for subplot rows. Default: None (aggregated).
        animation_frame
            Dimension for animation. Default: None (aggregated).
        **px_kwargs
            Additional arguments passed to `plotly.express.box()`.

        Returns
        -------
        plotly.graph_objects.Figure
        """
        return plotting.box(
            self._da,
            x=x,
            color=color,
            facet_col=facet_col,
            facet_row=facet_row,
            animation_frame=animation_frame,
            **px_kwargs,
        )

    def imshow(
        self,
        *,
        x: SlotValue = auto,
        y: SlotValue = auto,
        facet_col: SlotValue = auto,
        animation_frame: SlotValue = auto,
        **px_kwargs: Any,
    ) -> go.Figure:
        """
        Create an interactive heatmap image using Plotly Express.

        Dimensions are assigned to plot slots by their order:
        y (rows) -> x (columns) -> facet_col -> animation_frame

        Parameters
        ----------
        x
            Dimension for x-axis (columns). Default: second dimension.
        y
            Dimension for y-axis (rows). Default: first dimension.
        facet_col
            Dimension for subplot columns. Default: third dimension.
        animation_frame
            Dimension for animation. Default: fourth dimension.
        **px_kwargs
            Additional arguments passed to `plotly.express.imshow()`.

        Returns
        -------
        plotly.graph_objects.Figure
        """
        return plotting.imshow(
            self._da,
            x=x,
            y=y,
            facet_col=facet_col,
            animation_frame=animation_frame,
            **px_kwargs,
        )
