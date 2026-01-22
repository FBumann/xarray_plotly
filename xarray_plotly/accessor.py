"""Accessor classes for Plotly Express plotting on DataArray and Dataset."""

from typing import Any, ClassVar

import plotly.graph_objects as go
from xarray import DataArray, Dataset

from xarray_plotly import plotting
from xarray_plotly.common import SlotValue, auto
from xarray_plotly.config import _options


class DataArrayPlotlyAccessor:
    """Plotly Express plotting accessor for xarray DataArray.

    Dimensions are automatically assigned to plot slots by position.
    All methods return Plotly Figure objects for interactive visualization.

    Available methods: line, bar, area, scatter, box, imshow

    Args:
        darray: The DataArray to plot.

    Example:
        ```python
        import xarray as xr
        import numpy as np

        da = xr.DataArray(np.random.rand(10, 3), dims=["time", "city"])
        fig = da.plotly.line()  # Auto: time->x, city->color
        fig = da.plotly.line(color="time", x="city")  # Explicit
        fig = da.plotly.line(color=None)  # Skip slot
        fig.update_layout(title="My Plot")  # Customize
        ```
    """

    __all__: ClassVar = ["line", "bar", "fast_bar", "area", "scatter", "box", "imshow", "pie"]

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
        """Create an interactive line plot.

        Slot order: x -> color -> line_dash -> symbol -> facet_col -> facet_row -> animation_frame

        Args:
            x: Dimension for x-axis. Default: first dimension.
            color: Dimension for color grouping. Default: second dimension.
            line_dash: Dimension for line dash style. Default: third dimension.
            symbol: Dimension for marker symbol. Default: fourth dimension.
            facet_col: Dimension for subplot columns. Default: fifth dimension.
            facet_row: Dimension for subplot rows. Default: sixth dimension.
            animation_frame: Dimension for animation. Default: seventh dimension.
            **px_kwargs: Additional arguments passed to `plotly.express.line()`.

        Returns:
            Interactive Plotly Figure.
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
        """Create an interactive bar chart.

        Slot order: x -> color -> pattern_shape -> facet_col -> facet_row -> animation_frame

        Args:
            x: Dimension for x-axis. Default: first dimension.
            color: Dimension for color grouping. Default: second dimension.
            pattern_shape: Dimension for bar fill pattern. Default: third dimension.
            facet_col: Dimension for subplot columns. Default: fourth dimension.
            facet_row: Dimension for subplot rows. Default: fifth dimension.
            animation_frame: Dimension for animation. Default: sixth dimension.
            **px_kwargs: Additional arguments passed to `plotly.express.bar()`.

        Returns:
            Interactive Plotly Figure.
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
        """Create an interactive stacked area chart.

        Slot order: x -> color -> pattern_shape -> facet_col -> facet_row -> animation_frame

        Args:
            x: Dimension for x-axis. Default: first dimension.
            color: Dimension for color/stacking. Default: second dimension.
            pattern_shape: Dimension for fill pattern. Default: third dimension.
            facet_col: Dimension for subplot columns. Default: fourth dimension.
            facet_row: Dimension for subplot rows. Default: fifth dimension.
            animation_frame: Dimension for animation. Default: sixth dimension.
            **px_kwargs: Additional arguments passed to `plotly.express.area()`.

        Returns:
            Interactive Plotly Figure.
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

    def fast_bar(
        self,
        *,
        x: SlotValue = auto,
        color: SlotValue = auto,
        facet_col: SlotValue = auto,
        facet_row: SlotValue = auto,
        animation_frame: SlotValue = auto,
        **px_kwargs: Any,
    ) -> go.Figure:
        """Create a bar-like chart using stacked areas for better performance.

        Slot order: x -> color -> facet_col -> facet_row -> animation_frame

        Args:
            x: Dimension for x-axis. Default: first dimension.
            color: Dimension for color/stacking. Default: second dimension.
            facet_col: Dimension for subplot columns. Default: third dimension.
            facet_row: Dimension for subplot rows. Default: fourth dimension.
            animation_frame: Dimension for animation. Default: fifth dimension.
            **px_kwargs: Additional arguments passed to `plotly.express.area()`.

        Returns:
            Interactive Plotly Figure.
        """
        return plotting.fast_bar(
            self._da,
            x=x,
            color=color,
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
        """Create an interactive scatter plot.

        By default, y-axis shows the DataArray values. Set y to a dimension
        name to create dimension-vs-dimension plots (e.g., lat vs lon).

        Slot order: x -> color -> symbol -> facet_col -> facet_row -> animation_frame

        Args:
            x: Dimension for x-axis. Default: first dimension.
            y: What to plot on y-axis. Default "value" uses DataArray values.
                Can be a dimension name for dimension vs dimension plots.
            color: Dimension for color grouping, or "value" for DataArray values.
            symbol: Dimension for marker symbol. Default: third dimension.
            facet_col: Dimension for subplot columns. Default: fourth dimension.
            facet_row: Dimension for subplot rows. Default: fifth dimension.
            animation_frame: Dimension for animation. Default: sixth dimension.
            **px_kwargs: Additional arguments passed to `plotly.express.scatter()`.

        Returns:
            Interactive Plotly Figure.
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
        """Create an interactive box plot.

        By default, only the first dimension is assigned to x; all other
        dimensions are aggregated into the box statistics.

        Slot order: x -> color -> facet_col -> facet_row -> animation_frame

        Args:
            x: Dimension for x-axis categories. Default: first dimension.
            color: Dimension for color grouping. Default: None (aggregated).
            facet_col: Dimension for subplot columns. Default: None (aggregated).
            facet_row: Dimension for subplot rows. Default: None (aggregated).
            animation_frame: Dimension for animation. Default: None (aggregated).
            **px_kwargs: Additional arguments passed to `plotly.express.box()`.

        Returns:
            Interactive Plotly Figure.
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
        robust: bool = False,
        **px_kwargs: Any,
    ) -> go.Figure:
        """Create an interactive heatmap image.

        Slot order: y (rows) -> x (columns) -> facet_col -> animation_frame

        Note:
            **Difference from px.imshow**: Color bounds are computed from the
            entire dataset by default, ensuring consistent coloring across
            animation frames. Use `zmin`/`zmax` to override.

        Args:
            x: Dimension for x-axis (columns). Default: second dimension.
            y: Dimension for y-axis (rows). Default: first dimension.
            facet_col: Dimension for subplot columns. Default: third dimension.
            animation_frame: Dimension for animation. Default: fourth dimension.
            robust: If True, use 2nd/98th percentiles for color bounds (handles outliers).
            **px_kwargs: Additional arguments passed to `plotly.express.imshow()`.
                Use `zmin` and `zmax` to manually set color scale bounds.

        Returns:
            Interactive Plotly Figure.
        """
        return plotting.imshow(
            self._da,
            x=x,
            y=y,
            facet_col=facet_col,
            animation_frame=animation_frame,
            robust=robust,
            **px_kwargs,
        )

    def pie(
        self,
        *,
        names: SlotValue = auto,
        color: SlotValue = None,
        facet_col: SlotValue = auto,
        facet_row: SlotValue = auto,
        **px_kwargs: Any,
    ) -> go.Figure:
        """Create an interactive pie chart.

        Slot order: names -> facet_col -> facet_row

        Args:
            names: Dimension for pie slice names/categories. Default: first dimension.
            color: Dimension for color grouping. Default: None (uses names).
            facet_col: Dimension for subplot columns. Default: second dimension.
            facet_row: Dimension for subplot rows. Default: third dimension.
            **px_kwargs: Additional arguments passed to `plotly.express.pie()`.

        Returns:
            Interactive Plotly Figure.
        """
        return plotting.pie(
            self._da,
            names=names,
            color=color,
            facet_col=facet_col,
            facet_row=facet_row,
            **px_kwargs,
        )


class DatasetPlotlyAccessor:
    """Plotly Express plotting accessor for xarray Dataset.

    Plot a single variable or all variables at once. When plotting all variables,
    a "variable" dimension is created that can be assigned to any slot.

    Available methods: line, bar, area, scatter, box

    Args:
        dataset: The Dataset to plot.

    Example:
        ```python
        import xarray as xr
        import numpy as np

        ds = xr.Dataset({
            "temperature": (["time", "city"], np.random.rand(10, 3)),
            "humidity": (["time", "city"], np.random.rand(10, 3)),
        })

        # Plot all variables - "variable" dimension auto-assigned to color
        fig = ds.plotly.line()

        # Control where "variable" goes
        fig = ds.plotly.line(facet_col="variable")

        # Plot single variable
        fig = ds.plotly.line(var="temperature")
        ```
    """

    __all__: ClassVar = ["line", "bar", "fast_bar", "area", "scatter", "box", "pie"]

    def __init__(self, dataset: Dataset) -> None:
        self._ds = dataset

    def __dir__(self) -> list[str]:
        """List available plot methods."""
        return list(self.__all__) + list(super().__dir__())

    def _get_dataarray(self, var: str | None) -> DataArray:
        """Get DataArray from Dataset, either single var or all via to_array().

        When combining all variables, "variable" is placed at the position
        specified by config.dataset_variable_position (default 1, second position).
        Supports Python-style negative indexing: -1 = last, -2 = second-to-last, etc.
        """
        if var is None:
            da = self._ds.to_array(dim="variable")
            pos = _options.dataset_variable_position
            # Move "variable" to configured position
            if len(da.dims) > 1 and pos != 0:
                dims = list(da.dims)
                dims.remove("variable")
                # Use Python-style indexing (handles negative indices correctly)
                # Clamp to valid range: -1 -> last, -2 -> second-to-last, etc.
                n = len(dims)
                insert_pos = max(0, n + pos + 1) if pos < 0 else min(pos, n)
                dims.insert(insert_pos, "variable")
                da = da.transpose(*dims)
            return da
        return self._ds[var]

    def line(
        self,
        var: str | None = None,
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
        """Create an interactive line plot.

        Args:
            var: Variable to plot. If None, plots all variables with "variable" dimension.
            x: Dimension for x-axis.
            color: Dimension for color grouping.
            line_dash: Dimension for line dash style.
            symbol: Dimension for marker symbol.
            facet_col: Dimension for subplot columns.
            facet_row: Dimension for subplot rows.
            animation_frame: Dimension for animation.
            **px_kwargs: Additional arguments passed to `plotly.express.line()`.

        Returns:
            Interactive Plotly Figure.
        """
        da = self._get_dataarray(var)
        return plotting.line(
            da,
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
        var: str | None = None,
        *,
        x: SlotValue = auto,
        color: SlotValue = auto,
        pattern_shape: SlotValue = auto,
        facet_col: SlotValue = auto,
        facet_row: SlotValue = auto,
        animation_frame: SlotValue = auto,
        **px_kwargs: Any,
    ) -> go.Figure:
        """Create an interactive bar chart.

        Args:
            var: Variable to plot. If None, plots all variables with "variable" dimension.
            x: Dimension for x-axis.
            color: Dimension for color grouping.
            pattern_shape: Dimension for bar fill pattern.
            facet_col: Dimension for subplot columns.
            facet_row: Dimension for subplot rows.
            animation_frame: Dimension for animation.
            **px_kwargs: Additional arguments passed to `plotly.express.bar()`.

        Returns:
            Interactive Plotly Figure.
        """
        da = self._get_dataarray(var)
        return plotting.bar(
            da,
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
        var: str | None = None,
        *,
        x: SlotValue = auto,
        color: SlotValue = auto,
        pattern_shape: SlotValue = auto,
        facet_col: SlotValue = auto,
        facet_row: SlotValue = auto,
        animation_frame: SlotValue = auto,
        **px_kwargs: Any,
    ) -> go.Figure:
        """Create an interactive stacked area chart.

        Args:
            var: Variable to plot. If None, plots all variables with "variable" dimension.
            x: Dimension for x-axis.
            color: Dimension for color/stacking.
            pattern_shape: Dimension for fill pattern.
            facet_col: Dimension for subplot columns.
            facet_row: Dimension for subplot rows.
            animation_frame: Dimension for animation.
            **px_kwargs: Additional arguments passed to `plotly.express.area()`.

        Returns:
            Interactive Plotly Figure.
        """
        da = self._get_dataarray(var)
        return plotting.area(
            da,
            x=x,
            color=color,
            pattern_shape=pattern_shape,
            facet_col=facet_col,
            facet_row=facet_row,
            animation_frame=animation_frame,
            **px_kwargs,
        )

    def fast_bar(
        self,
        var: str | None = None,
        *,
        x: SlotValue = auto,
        color: SlotValue = auto,
        facet_col: SlotValue = auto,
        facet_row: SlotValue = auto,
        animation_frame: SlotValue = auto,
        **px_kwargs: Any,
    ) -> go.Figure:
        """Create a bar-like chart using stacked areas for better performance.

        Args:
            var: Variable to plot. If None, plots all variables with "variable" dimension.
            x: Dimension for x-axis.
            color: Dimension for color/stacking.
            facet_col: Dimension for subplot columns.
            facet_row: Dimension for subplot rows.
            animation_frame: Dimension for animation.
            **px_kwargs: Additional arguments passed to `plotly.express.area()`.

        Returns:
            Interactive Plotly Figure.
        """
        da = self._get_dataarray(var)
        return plotting.fast_bar(
            da,
            x=x,
            color=color,
            facet_col=facet_col,
            facet_row=facet_row,
            animation_frame=animation_frame,
            **px_kwargs,
        )

    def scatter(
        self,
        var: str | None = None,
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
        """Create an interactive scatter plot.

        Args:
            var: Variable to plot. If None, plots all variables with "variable" dimension.
            x: Dimension for x-axis.
            y: What to plot on y-axis. Default "value" uses DataArray values.
            color: Dimension for color grouping, or "value" for DataArray values.
            symbol: Dimension for marker symbol.
            facet_col: Dimension for subplot columns.
            facet_row: Dimension for subplot rows.
            animation_frame: Dimension for animation.
            **px_kwargs: Additional arguments passed to `plotly.express.scatter()`.

        Returns:
            Interactive Plotly Figure.
        """
        da = self._get_dataarray(var)
        return plotting.scatter(
            da,
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
        var: str | None = None,
        *,
        x: SlotValue = auto,
        color: SlotValue = None,
        facet_col: SlotValue = None,
        facet_row: SlotValue = None,
        animation_frame: SlotValue = None,
        **px_kwargs: Any,
    ) -> go.Figure:
        """Create an interactive box plot.

        Args:
            var: Variable to plot. If None, plots all variables with "variable" dimension.
            x: Dimension for x-axis categories.
            color: Dimension for color grouping.
            facet_col: Dimension for subplot columns.
            facet_row: Dimension for subplot rows.
            animation_frame: Dimension for animation.
            **px_kwargs: Additional arguments passed to `plotly.express.box()`.

        Returns:
            Interactive Plotly Figure.
        """
        da = self._get_dataarray(var)
        return plotting.box(
            da,
            x=x,
            color=color,
            facet_col=facet_col,
            facet_row=facet_row,
            animation_frame=animation_frame,
            **px_kwargs,
        )

    def pie(
        self,
        var: str | None = None,
        *,
        names: SlotValue = auto,
        color: SlotValue = None,
        facet_col: SlotValue = auto,
        facet_row: SlotValue = auto,
        **px_kwargs: Any,
    ) -> go.Figure:
        """Create an interactive pie chart.

        Args:
            var: Variable to plot. If None, plots all variables with "variable" dimension.
            names: Dimension for pie slice names/categories.
            color: Dimension for color grouping.
            facet_col: Dimension for subplot columns.
            facet_row: Dimension for subplot rows.
            **px_kwargs: Additional arguments passed to `plotly.express.pie()`.

        Returns:
            Interactive Plotly Figure.
        """
        da = self._get_dataarray(var)
        return plotting.pie(
            da,
            names=names,
            color=color,
            facet_col=facet_col,
            facet_row=facet_row,
            **px_kwargs,
        )
