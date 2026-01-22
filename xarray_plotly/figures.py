"""
Helper functions for combining and manipulating Plotly figures.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

    import plotly.graph_objects as go


def _iter_all_traces(fig: go.Figure) -> Iterator:
    """Iterate over all traces in a figure, including animation frames.

    Yields traces from fig.data first, then from each frame in fig.frames.
    Useful for applying styling to all traces including those in animations.

    Args:
        fig: Plotly Figure.

    Yields:
        Each trace object from the figure.
    """
    yield from fig.data
    for frame in fig.frames or []:
        yield from frame.data


def _get_subplot_axes(fig: go.Figure) -> set[tuple[str, str]]:
    """Extract (xaxis, yaxis) pairs from figure traces.

    Args:
        fig: A Plotly figure.

    Returns:
        Set of (xaxis, yaxis) tuples, e.g., {('x', 'y'), ('x2', 'y2')}.
    """
    axes_pairs = set()
    for trace in fig.data:
        xaxis = getattr(trace, "xaxis", None) or "x"
        yaxis = getattr(trace, "yaxis", None) or "y"
        axes_pairs.add((xaxis, yaxis))
    return axes_pairs


def _validate_compatible_structure(base: go.Figure, overlay: go.Figure) -> None:
    """Validate that overlay's subplot structure is compatible with base.

    Args:
        base: The base figure.
        overlay: The overlay figure to check.

    Raises:
        ValueError: If overlay has subplots not present in base.
    """
    base_axes = _get_subplot_axes(base)
    overlay_axes = _get_subplot_axes(overlay)

    extra_axes = overlay_axes - base_axes
    if extra_axes:
        raise ValueError(
            f"Overlay figure has subplots not present in base figure: {extra_axes}. "
            "Ensure both figures have the same facet structure."
        )


def _validate_animation_compatibility(base: go.Figure, overlay: go.Figure) -> None:
    """Validate animation frame compatibility between base and overlay.

    Args:
        base: The base figure.
        overlay: The overlay figure to check.

    Raises:
        ValueError: If overlay has animation but base doesn't, or frame names don't match.
    """
    base_has_frames = bool(base.frames)
    overlay_has_frames = bool(overlay.frames)

    if overlay_has_frames and not base_has_frames:
        raise ValueError(
            "Overlay figure has animation frames but base figure does not. "
            "Cannot add animated overlay to static base figure."
        )

    if base_has_frames and overlay_has_frames:
        base_frame_names = {frame.name for frame in base.frames}
        overlay_frame_names = {frame.name for frame in overlay.frames}

        if base_frame_names != overlay_frame_names:
            missing_in_overlay = base_frame_names - overlay_frame_names
            extra_in_overlay = overlay_frame_names - base_frame_names
            msg = "Animation frame names don't match between base and overlay."
            if missing_in_overlay:
                msg += f" Missing in overlay: {missing_in_overlay}."
            if extra_in_overlay:
                msg += f" Extra in overlay: {extra_in_overlay}."
            raise ValueError(msg)


def _merge_frames(
    base: go.Figure,
    overlays: list[go.Figure],
    base_trace_count: int,
    overlay_trace_counts: list[int],
) -> list:
    """Merge animation frames from base and overlay figures.

    Args:
        base: The base figure with animation frames.
        overlays: List of overlay figures (may or may not have frames).
        base_trace_count: Number of traces in the base figure.
        overlay_trace_counts: Number of traces in each overlay figure.

    Returns:
        List of merged frames.
    """
    import plotly.graph_objects as go

    merged_frames = []

    for base_frame in base.frames:
        frame_name = base_frame.name
        merged_data = list(base_frame.data)

        for overlay, _overlay_trace_count in zip(overlays, overlay_trace_counts, strict=False):
            if overlay.frames:
                # Find matching frame in overlay
                overlay_frame = next((f for f in overlay.frames if f.name == frame_name), None)
                if overlay_frame:
                    merged_data.extend(overlay_frame.data)
            else:
                # Static overlay: replicate traces to this frame
                merged_data.extend(overlay.data)

        merged_frames.append(
            go.Frame(
                data=merged_data,
                name=frame_name,
                traces=list(range(base_trace_count + sum(overlay_trace_counts))),
            )
        )

    return merged_frames


def overlay(base: go.Figure, *overlays: go.Figure) -> go.Figure:
    """Overlay multiple Plotly figures on the same axes.

    Creates a new figure with the base figure's layout, sliders, and buttons,
    with all overlay traces added on top. Correctly handles faceted figures
    and animation frames.

    Args:
        base: The base figure whose layout is preserved.
        *overlays: One or more figures to overlay on the base.

    Returns:
        A new combined figure.

    Raises:
        ValueError: If overlay has subplots not in base, animation frames don't match,
            or overlay has animation but base doesn't.

    Example:
        >>> import numpy as np
        >>> import xarray as xr
        >>> from xarray_plotly import xpx, overlay
        >>>
        >>> da = xr.DataArray(np.random.rand(10, 3), dims=["time", "cat"])
        >>> area_fig = xpx(da).area()
        >>> line_fig = xpx(da).line()
        >>> combined = overlay(area_fig, line_fig)
        >>>
        >>> # With animation
        >>> da3d = xr.DataArray(np.random.rand(10, 3, 4), dims=["x", "cat", "time"])
        >>> area = xpx(da3d).area(animation_frame="time")
        >>> line = xpx(da3d).line(animation_frame="time")
        >>> combined = overlay(area, line)
    """
    import plotly.graph_objects as go

    if not overlays:
        # No overlays: return a deep copy of base
        return copy.deepcopy(base)

    # Validate all overlays
    for overlay in overlays:
        _validate_compatible_structure(base, overlay)
        _validate_animation_compatibility(base, overlay)

    # Create new figure with base's layout
    combined = go.Figure(layout=copy.deepcopy(base.layout))

    # Add all traces from base
    for trace in base.data:
        combined.add_trace(copy.deepcopy(trace))

    # Add all traces from overlays
    for overlay in overlays:
        for trace in overlay.data:
            combined.add_trace(copy.deepcopy(trace))

    # Handle animation frames
    if base.frames:
        base_trace_count = len(base.data)
        overlay_trace_counts = [len(overlay.data) for overlay in overlays]
        merged_frames = _merge_frames(base, list(overlays), base_trace_count, overlay_trace_counts)
        combined.frames = merged_frames

    return combined


def _build_secondary_y_mapping(base_axes: set[tuple[str, str]]) -> dict[str, str]:
    """Build mapping from primary y-axes to secondary y-axes.

    Args:
        base_axes: Set of (xaxis, yaxis) pairs from base figure.

    Returns:
        Dict mapping primary yaxis names to secondary yaxis names.
        E.g., {'y': 'y4', 'y2': 'y5', 'y3': 'y6'}
    """
    primary_y_axes = sorted({yaxis for _, yaxis in base_axes})

    # Find the highest existing yaxis number
    max_y_num = 1  # 'y' is 1
    for yaxis in primary_y_axes:
        num = 1 if yaxis == "y" else int(yaxis[1:])
        max_y_num = max(max_y_num, num)

    # Create mapping: primary_yaxis -> secondary_yaxis
    y_mapping = {}
    next_y_num = max_y_num + 1
    for yaxis in primary_y_axes:
        y_mapping[yaxis] = f"y{next_y_num}"
        next_y_num += 1

    return y_mapping


def add_secondary_y(
    base: go.Figure,
    secondary: go.Figure,
    *,
    secondary_y_title: str | None = None,
) -> go.Figure:
    """Add a secondary y-axis with traces from another figure.

    Creates a new figure with the base figure's layout and secondary y-axes
    on the right side. All traces from the secondary figure are plotted against
    the secondary y-axes. Supports faceted figures when both have matching
    facet structure.

    Args:
        base: The base figure (left y-axis).
        secondary: The figure whose traces use the secondary y-axis (right).
        secondary_y_title: Optional title for the secondary y-axis.
            If not provided, uses the secondary figure's y-axis title.

    Returns:
        A new figure with both primary and secondary y-axes.

    Raises:
        ValueError: If facet structures don't match, or if animation
            frames don't match.

    Example:
        >>> import numpy as np
        >>> import xarray as xr
        >>> from xarray_plotly import xpx, add_secondary_y
        >>>
        >>> # Two variables with different scales
        >>> temp = xr.DataArray([20, 22, 25, 23], dims=["time"], name="Temperature (°C)")
        >>> precip = xr.DataArray([0, 5, 12, 2], dims=["time"], name="Precipitation (mm)")
        >>>
        >>> temp_fig = xpx(temp).line()
        >>> precip_fig = xpx(precip).bar()
        >>> combined = add_secondary_y(temp_fig, precip_fig)
        >>>
        >>> # With facets
        >>> data = xr.DataArray(np.random.rand(10, 3), dims=["x", "facet"])
        >>> fig1 = xpx(data).line(facet_col="facet")
        >>> fig2 = xpx(data * 100).bar(facet_col="facet")  # Different scale
        >>> combined = add_secondary_y(fig1, fig2)
    """
    import plotly.graph_objects as go

    # Get axis pairs from both figures
    base_axes = _get_subplot_axes(base)
    secondary_axes = _get_subplot_axes(secondary)

    # Validate same facet structure
    if base_axes != secondary_axes:
        raise ValueError(
            f"Base and secondary figures must have the same facet structure. "
            f"Base has {base_axes}, secondary has {secondary_axes}."
        )

    # Validate animation compatibility
    _validate_animation_compatibility(base, secondary)

    # Build mapping from primary y-axes to secondary y-axes
    y_mapping = _build_secondary_y_mapping(base_axes)

    # Create new figure with base's layout
    combined = go.Figure(layout=copy.deepcopy(base.layout))

    # Add all traces from base (primary y-axis)
    for trace in base.data:
        combined.add_trace(copy.deepcopy(trace))

    # Add all traces from secondary, remapped to secondary y-axes
    for trace in secondary.data:
        trace_copy = copy.deepcopy(trace)
        original_yaxis = getattr(trace_copy, "yaxis", None) or "y"
        trace_copy.yaxis = y_mapping[original_yaxis]
        combined.add_trace(trace_copy)

    # Configure secondary y-axes
    for primary_yaxis, secondary_yaxis in y_mapping.items():
        # Get title - only set on first secondary axis or use provided title
        title = None
        if secondary_y_title is not None:
            # Only set title on the first secondary axis to avoid repetition
            if primary_yaxis == "y":
                title = secondary_y_title
        elif primary_yaxis == "y" and secondary.layout.yaxis and secondary.layout.yaxis.title:
            # Try to get from secondary's layout
            title = secondary.layout.yaxis.title.text

        # Configure the secondary axis
        axis_config = {
            "title": title,
            "overlaying": primary_yaxis,
            "side": "right",
            "anchor": "free" if primary_yaxis != "y" else None,
        }
        # Remove None values
        axis_config = {k: v for k, v in axis_config.items() if v is not None}

        # Convert y2 -> yaxis2, y3 -> yaxis3, etc. for layout property name
        layout_prop = "yaxis" if secondary_yaxis == "y" else f"yaxis{secondary_yaxis[1:]}"
        combined.update_layout(**{layout_prop: axis_config})

    # Handle animation frames
    if base.frames:
        merged_frames = _merge_secondary_y_frames(base, secondary, y_mapping)
        combined.frames = merged_frames

    return combined


def _merge_secondary_y_frames(
    base: go.Figure,
    secondary: go.Figure,
    y_mapping: dict[str, str],
) -> list:
    """Merge animation frames for secondary y-axis combination.

    Args:
        base: The base figure with animation frames.
        secondary: The secondary figure (may or may not have frames).
        y_mapping: Mapping from primary y-axis names to secondary y-axis names.

    Returns:
        List of merged frames with secondary traces assigned to secondary y-axes.
    """
    import plotly.graph_objects as go

    merged_frames = []
    base_trace_count = len(base.data)
    secondary_trace_count = len(secondary.data)

    for base_frame in base.frames:
        frame_name = base_frame.name
        merged_data = list(base_frame.data)

        if secondary.frames:
            # Find matching frame in secondary
            secondary_frame = next((f for f in secondary.frames if f.name == frame_name), None)
            if secondary_frame:
                # Add secondary frame data with remapped y-axis
                for trace_data in secondary_frame.data:
                    trace_copy = copy.deepcopy(trace_data)
                    original_yaxis = getattr(trace_copy, "yaxis", None) or "y"
                    trace_copy.yaxis = y_mapping.get(original_yaxis, original_yaxis)
                    merged_data.append(trace_copy)
        else:
            # Static secondary: replicate traces to this frame
            for trace in secondary.data:
                trace_copy = copy.deepcopy(trace)
                original_yaxis = getattr(trace_copy, "yaxis", None) or "y"
                trace_copy.yaxis = y_mapping.get(original_yaxis, original_yaxis)
                merged_data.append(trace_copy)

        merged_frames.append(
            go.Frame(
                data=merged_data,
                name=frame_name,
                traces=list(range(base_trace_count + secondary_trace_count)),
            )
        )

    return merged_frames


def update_traces(fig: go.Figure, selector: dict | None = None, **kwargs) -> go.Figure:
    """Update traces in both base figure and all animation frames.

    Plotly's `update_traces()` only updates the base figure, not animation frames.
    This function updates both, ensuring trace styles persist during animation.

    Args:
        fig: A Plotly figure, optionally with animation frames.
        selector: Dict to match specific traces, e.g. ``{"name": "Germany"}``.
            If None, updates all traces.
        **kwargs: Trace properties to update, e.g. ``line_width=4``, ``line_dash="dot"``.

    Returns:
        The modified figure (same object, mutated in place).

    Example:
        >>> import plotly.express as px
        >>> from xarray_plotly import update_traces
        >>>
        >>> df = px.data.gapminder()
        >>> fig = px.line(df, x="year", y="gdpPercap", color="country", animation_frame="continent")
        >>>
        >>> # Update all traces
        >>> update_traces(fig, line_width=3)
        >>>
        >>> # Update specific trace by name
        >>> update_traces(fig, selector={"name": "Germany"}, line_width=5, line_dash="dot")
    """
    for trace in _iter_all_traces(fig):
        if selector is None:
            trace.update(**kwargs)
        else:
            # Check if trace matches all selector criteria
            if all(getattr(trace, k, None) == v for k, v in selector.items()):
                trace.update(**kwargs)

    return fig
