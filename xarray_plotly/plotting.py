"""
Plotly Express plotting functions for DataArray objects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import plotly.express as px

from xarray_plotly.common import (
    SlotValue,
    assign_slots,
    auto,
    build_labels,
    get_label,
    get_value_col,
    to_dataframe,
)

if TYPE_CHECKING:
    import plotly.graph_objects as go
    from xarray import DataArray


def line(
    darray: DataArray,
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
    Create an interactive line plot from a DataArray.

    The y-axis shows DataArray values. Dimensions fill slots in order:
    x -> color -> line_dash -> symbol -> facet_col -> facet_row -> animation_frame

    Parameters
    ----------
    darray
        The DataArray to plot.
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
    slots = assign_slots(
        list(darray.dims),
        "line",
        x=x,
        color=color,
        line_dash=line_dash,
        symbol=symbol,
        facet_col=facet_col,
        facet_row=facet_row,
        animation_frame=animation_frame,
    )

    df = to_dataframe(darray)
    value_col = get_value_col(darray)
    labels = {**build_labels(darray, slots, value_col), **px_kwargs.pop("labels", {})}

    return px.line(
        df,
        x=slots.get("x"),
        y=value_col,
        color=slots.get("color"),
        line_dash=slots.get("line_dash"),
        symbol=slots.get("symbol"),
        facet_col=slots.get("facet_col"),
        facet_row=slots.get("facet_row"),
        animation_frame=slots.get("animation_frame"),
        labels=labels,
        **px_kwargs,
    )


def bar(
    darray: DataArray,
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
    Create an interactive bar chart from a DataArray.

    The y-axis shows DataArray values. Dimensions fill slots in order:
    x -> color -> pattern_shape -> facet_col -> facet_row -> animation_frame

    Parameters
    ----------
    darray
        The DataArray to plot.
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
    slots = assign_slots(
        list(darray.dims),
        "bar",
        x=x,
        color=color,
        pattern_shape=pattern_shape,
        facet_col=facet_col,
        facet_row=facet_row,
        animation_frame=animation_frame,
    )

    df = to_dataframe(darray)
    value_col = get_value_col(darray)
    labels = {**build_labels(darray, slots, value_col), **px_kwargs.pop("labels", {})}

    return px.bar(
        df,
        x=slots.get("x"),
        y=value_col,
        color=slots.get("color"),
        pattern_shape=slots.get("pattern_shape"),
        facet_col=slots.get("facet_col"),
        facet_row=slots.get("facet_row"),
        animation_frame=slots.get("animation_frame"),
        labels=labels,
        **px_kwargs,
    )


def area(
    darray: DataArray,
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
    Create an interactive stacked area chart from a DataArray.

    The y-axis shows DataArray values. Dimensions fill slots in order:
    x -> color -> pattern_shape -> facet_col -> facet_row -> animation_frame

    Parameters
    ----------
    darray
        The DataArray to plot.
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
    slots = assign_slots(
        list(darray.dims),
        "area",
        x=x,
        color=color,
        pattern_shape=pattern_shape,
        facet_col=facet_col,
        facet_row=facet_row,
        animation_frame=animation_frame,
    )

    df = to_dataframe(darray)
    value_col = get_value_col(darray)
    labels = {**build_labels(darray, slots, value_col), **px_kwargs.pop("labels", {})}

    return px.area(
        df,
        x=slots.get("x"),
        y=value_col,
        color=slots.get("color"),
        pattern_shape=slots.get("pattern_shape"),
        facet_col=slots.get("facet_col"),
        facet_row=slots.get("facet_row"),
        animation_frame=slots.get("animation_frame"),
        labels=labels,
        **px_kwargs,
    )


def box(
    darray: DataArray,
    *,
    x: SlotValue = auto,
    color: SlotValue = None,
    facet_col: SlotValue = None,
    facet_row: SlotValue = None,
    animation_frame: SlotValue = None,
    **px_kwargs: Any,
) -> go.Figure:
    """
    Create an interactive box plot from a DataArray.

    The y-axis shows DataArray values. By default, only x is auto-assigned;
    other dimensions are aggregated into the box statistics.

    Dimensions fill slots in order: x -> color -> facet_col -> facet_row -> animation_frame

    Parameters
    ----------
    darray
        The DataArray to plot.
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
    slots = assign_slots(
        list(darray.dims),
        "box",
        allow_unassigned=True,
        x=x,
        color=color,
        facet_col=facet_col,
        facet_row=facet_row,
        animation_frame=animation_frame,
    )

    df = to_dataframe(darray)
    value_col = get_value_col(darray)
    labels = {**build_labels(darray, slots, value_col), **px_kwargs.pop("labels", {})}

    return px.box(
        df,
        x=slots.get("x"),
        y=value_col,
        color=slots.get("color"),
        facet_col=slots.get("facet_col"),
        facet_row=slots.get("facet_row"),
        animation_frame=slots.get("animation_frame"),
        labels=labels,
        **px_kwargs,
    )


def scatter(
    darray: DataArray,
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
    Create an interactive scatter plot from a DataArray.

    By default, y-axis shows DataArray values. Set y to a dimension name
    for dimension-vs-dimension plots (e.g., lat vs lon colored by value).

    Dimensions fill slots in order:
    x -> color -> symbol -> facet_col -> facet_row -> animation_frame

    Parameters
    ----------
    darray
        The DataArray to plot.
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
    # If y is a dimension, exclude it from slot assignment
    y_is_dim = y != "value" and y in darray.dims
    dims_for_slots = [d for d in darray.dims if d != y] if y_is_dim else list(darray.dims)

    slots = assign_slots(
        dims_for_slots,
        "scatter",
        x=x,
        color=color,
        symbol=symbol,
        facet_col=facet_col,
        facet_row=facet_row,
        animation_frame=animation_frame,
    )

    df = to_dataframe(darray)
    value_col = get_value_col(darray)

    # Resolve y and color columns (may be "value" -> actual column name)
    y_col = value_col if y == "value" else y
    color_col = value_col if slots.get("color") == "value" else slots.get("color")

    # Build labels
    labels = {**build_labels(darray, slots, value_col), **px_kwargs.pop("labels", {})}
    if y_is_dim and str(y) not in labels:
        labels[str(y)] = get_label(darray, y)

    return px.scatter(
        df,
        x=slots.get("x"),
        y=y_col,
        color=color_col,
        symbol=slots.get("symbol"),
        facet_col=slots.get("facet_col"),
        facet_row=slots.get("facet_row"),
        animation_frame=slots.get("animation_frame"),
        labels=labels,
        **px_kwargs,
    )


def imshow(
    darray: DataArray,
    *,
    x: SlotValue = auto,
    y: SlotValue = auto,
    facet_col: SlotValue = auto,
    animation_frame: SlotValue = auto,
    **px_kwargs: Any,
) -> go.Figure:
    """
    Create an interactive heatmap from a DataArray.

    Both x and y are dimensions. Dimensions fill slots in order:
    y (rows) -> x (columns) -> facet_col -> animation_frame

    Parameters
    ----------
    darray
        The DataArray to plot.
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
    slots = assign_slots(
        list(darray.dims),
        "imshow",
        y=y,
        x=x,
        facet_col=facet_col,
        animation_frame=animation_frame,
    )

    # Transpose to: y (rows), x (cols), facet_col, animation_frame
    transpose_order = [
        slots[k] for k in ("y", "x", "facet_col", "animation_frame") if slots.get(k) is not None
    ]
    plot_data = darray.transpose(*transpose_order) if transpose_order else darray

    return px.imshow(
        plot_data,
        facet_col=slots.get("facet_col"),
        animation_frame=slots.get("animation_frame"),
        **px_kwargs,
    )
