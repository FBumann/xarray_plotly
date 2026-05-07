"""
Helper functions for combining and manipulating Plotly figures.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

    import plotly.graph_objects as go


def _get_yaxis_title(fig: go.Figure) -> str:
    """Extract the primary y-axis title text from a figure.

    Args:
        fig: A Plotly figure.

    Returns:
        The y-axis title text, or empty string if not set.
    """
    try:
        return fig.layout.yaxis.title.text or ""
    except AttributeError:
        return ""


def _dedup_legend_within_traces(traces: list[Any]) -> None:
    """Ensure one ``showlegend=True`` per ``legendgroup`` among the given traces."""
    from collections import defaultdict

    grouped: dict[str, list[Any]] = defaultdict(list)
    ungrouped: list[Any] = []

    for trace in traces:
        lg = getattr(trace, "legendgroup", None) or ""
        if lg:
            grouped[lg].append(trace)
        else:
            ungrouped.append(trace)

    for group_traces in grouped.values():
        has_visible = False
        for t in group_traces:
            if has_visible:
                t.showlegend = False
            elif getattr(t, "name", None):
                t.showlegend = True
                has_visible = True

    for trace in ungrouped:
        if getattr(trace, "name", None):
            trace.showlegend = True


def _ensure_legend_visibility(
    combined: go.Figure,
    source_figs: list[go.Figure],
    trace_slices: list[slice],
    *,
    cross_source_dedup: bool = True,
) -> None:
    """Fix legend visibility on a combined figure.

    Handles three problems that arise when combining Plotly Express figures:

    1. **Unnamed traces** — PX sets ``name=""`` on single-trace (no color)
       figures.  We derive a name from each source figure's y-axis title.
    2. **Hidden named traces** — PX sets ``showlegend=False`` on single-trace
       figures.  We ensure at least one trace per ``legendgroup`` (or each
       ungrouped named trace) has ``showlegend=True``.
    3. **Duplicate legend entries** — when two source figures share the same
       ``legendgroup`` names and ``cross_source_dedup=True`` (the default),
       we deduplicate so only the first trace per group shows in the legend.
       When ``cross_source_dedup=False``, traces from different sources are
       kept independent: colliding ``legendgroup`` names are namespaced with
       the source label so each source's traces get their own legend entries.

    Args:
        combined: The combined Plotly figure (mutated in place).
        source_figs: The original source figures, in trace order.
        trace_slices: Slices into ``combined.data`` for each source figure.
        cross_source_dedup: If True (overlay default), dedup legend entries
            across all sources. If False (add_secondary_y), preserve each
            source's legend entries independently.
    """
    from collections import defaultdict

    # --- Step 1: label unnamed traces from source y-axis titles -----------
    labels = [_get_yaxis_title(f) for f in source_figs]

    # If all labels are non-empty and identical, disambiguate
    unique_labels = {lb for lb in labels if lb}
    if len(unique_labels) == 1 and all(lb for lb in labels):
        labels = [f"{labels[0]} ({i + 1})" for i in range(len(labels))]

    for label, sl in zip(labels, trace_slices, strict=False):
        if not label:
            continue
        for trace in combined.data[sl]:
            if not getattr(trace, "name", None):
                trace.name = label
                trace.legendgroup = label

    # --- Step 2 & 3: fix showlegend per legendgroup -----------------------
    if cross_source_dedup:
        _dedup_legend_within_traces(list(combined.data))
    else:
        # Namespace legendgroups that collide across slices, so each source
        # keeps its own legend entries instead of being deduped away.
        slice_groups: list[set[str]] = []
        for sl in trace_slices:
            slice_groups.append(
                {
                    getattr(t, "legendgroup", None)
                    for t in combined.data[sl]
                    if getattr(t, "legendgroup", None)
                }  # type: ignore[misc]
            )
        group_counts: dict[str, int] = defaultdict(int)
        for sg in slice_groups:
            for g in sg:
                group_counts[g] += 1
        colliding = {g for g, cnt in group_counts.items() if cnt > 1}

        for label, sl in zip(labels, trace_slices, strict=False):
            if not label:
                continue
            for trace in combined.data[sl]:
                lg = getattr(trace, "legendgroup", None)
                if lg and lg in colliding:
                    new_lg = f"{lg} ({label})"
                    trace.legendgroup = new_lg
                    if getattr(trace, "name", None) == lg:
                        trace.name = new_lg

        for sl in trace_slices:
            _dedup_legend_within_traces(list(combined.data[sl]))

    # --- Step 4: propagate style properties to animation frame traces ------
    # When Plotly animates, frame trace data overwrites fig.data properties.
    # PX frame traces carry name="", showlegend=False and default colors,
    # discarding any styling the user applied via update_traces() before
    # combining.  Propagate display properties from fig.data into every frame.
    _STYLE_ATTRS = ("name", "legendgroup", "showlegend", "marker", "line", "opacity")
    for frame in combined.frames or []:
        for i, frame_trace in enumerate(frame.data):
            if i < len(combined.data):
                src = combined.data[i]
                for attr in _STYLE_ATTRS:
                    src_val = getattr(src, attr, None)
                    if src_val is not None:
                        setattr(frame_trace, attr, src_val)


def _fix_animation_axis_ranges(fig: go.Figure) -> None:
    """Set axis ranges to encompass data across all animation frames.

    Plotly.js computes autorange from ``fig.data`` only and does not
    recalculate during animation.  When different frames have very different
    data ranges (e.g. population of Brazil vs China), values can go off-screen.
    This function computes the global min/max for each axis across all frames
    and sets explicit ranges on the layout.

    Only numeric axes are handled; categorical/date axes are left to autorange.

    Args:
        fig: A Plotly figure with animation frames (mutated in place).
    """
    import numpy as np

    if not fig.frames:
        return

    from collections import defaultdict

    # Collect numeric y-values per axis across all traces (fig.data + frames)
    y_by_axis: dict[str, list[float]] = defaultdict(list)
    x_by_axis: dict[str, list[float]] = defaultdict(list)

    # Track which axes have bar traces (for zero-baseline clamping)
    y_has_vbar: set[str] = set()  # vertical bars → y-axis includes 0
    x_has_hbar: set[str] = set()  # horizontal bars → x-axis includes 0

    for trace in _iter_all_traces(fig):
        yaxis = getattr(trace, "yaxis", None) or "y"
        xaxis = getattr(trace, "xaxis", None) or "x"

        # Track bar orientations
        if getattr(trace, "type", None) == "bar":
            orientation = getattr(trace, "orientation", None) or "v"
            if orientation == "h":
                x_has_hbar.add(xaxis)
            else:
                y_has_vbar.add(yaxis)

        for data_attr, axis_ref, by_axis in [
            ("y", yaxis, y_by_axis),
            ("x", xaxis, x_by_axis),
        ]:
            vals = getattr(trace, data_attr, None)
            if vals is None:
                continue
            arr = np.asarray(vals)
            # Skip datetime/timedelta — leave those axes on autorange
            if np.issubdtype(arr.dtype, np.datetime64) or np.issubdtype(arr.dtype, np.timedelta64):
                continue
            try:
                arr = arr.astype(float)
                finite = arr[np.isfinite(arr)]
                if len(finite):
                    by_axis[axis_ref].extend(finite.tolist())
            except (ValueError, TypeError):
                pass  # Non-numeric (categorical) — skip

    # Apply ranges to layout
    for axis_ref, values in y_by_axis.items():
        if not values:
            continue
        lo, hi = min(values), max(values)
        if axis_ref in y_has_vbar:
            lo = min(lo, 0.0)
            hi = max(hi, 0.0)
        pad = (hi - lo) * 0.05 or 1  # 5% padding
        layout_prop = "yaxis" if axis_ref == "y" else f"yaxis{axis_ref[1:]}"
        fig.layout[layout_prop].range = [lo - pad, hi + pad]

    for axis_ref, values in x_by_axis.items():
        if not values:
            continue
        lo, hi = min(values), max(values)
        if axis_ref in x_has_hbar:
            lo = min(lo, 0.0)
            hi = max(hi, 0.0)
        pad = (hi - lo) * 0.05 or 1
        layout_prop = "xaxis" if axis_ref == "x" else f"xaxis{axis_ref[1:]}"
        fig.layout[layout_prop].range = [lo - pad, hi + pad]


def _iter_all_traces(fig: go.Figure) -> Iterator[Any]:
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
) -> list[go.Frame]:
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
                layout=base_frame.layout,
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

    # Create new figure with base's layout and all traces
    all_traces = [copy.deepcopy(t) for t in base.data]
    for overlay in overlays:
        all_traces.extend(copy.deepcopy(t) for t in overlay.data)
    combined = go.Figure(data=all_traces, layout=copy.deepcopy(base.layout))

    # Handle animation frames
    if base.frames:
        base_trace_count = len(base.data)
        overlay_trace_counts = [len(overlay.data) for overlay in overlays]
        merged_frames = _merge_frames(base, list(overlays), base_trace_count, overlay_trace_counts)
        combined.frames = merged_frames

    # Build trace slices for legend fix
    source_figs = [base, *overlays]
    slices: list[slice] = []
    offset = 0
    for fig in source_figs:
        n = len(fig.data)
        slices.append(slice(offset, offset + n))
        offset += n

    _ensure_legend_visibility(combined, source_figs, slices)
    _fix_animation_axis_ranges(combined)
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

    # Build x-y correspondence from base_axes (which x-axis pairs with which y-axis)
    x_for_y = {yaxis: xaxis for xaxis, yaxis in base_axes}

    # Find the rightmost x-axis (highest number) to determine which secondary axis shows ticks
    rightmost_x = max(x_for_y.values(), key=lambda x: int(x[1:]) if x != "x" else 1)
    rightmost_primary_y = next(y for y, x in x_for_y.items() if x == rightmost_x)

    # Build all traces: base (primary) + secondary (remapped to secondary y-axes)
    all_traces = [copy.deepcopy(t) for t in base.data]
    for trace in secondary.data:
        trace_copy = copy.deepcopy(trace)
        original_yaxis = getattr(trace_copy, "yaxis", None) or "y"
        trace_copy.yaxis = y_mapping[original_yaxis]
        all_traces.append(trace_copy)

    combined = go.Figure(data=all_traces, layout=copy.deepcopy(base.layout))

    # Get the rightmost secondary y-axis name for linking
    rightmost_secondary_y = y_mapping[rightmost_primary_y]

    # Configure secondary y-axes
    for primary_yaxis, secondary_yaxis in y_mapping.items():
        is_rightmost = primary_yaxis == rightmost_primary_y

        # Get title - only set on rightmost secondary axis
        title = None
        if is_rightmost:
            if secondary_y_title is not None:
                title = secondary_y_title
            elif secondary.layout.yaxis and secondary.layout.yaxis.title:
                title = secondary.layout.yaxis.title.text

        # Configure the secondary axis
        # Anchor to the corresponding x-axis so it appears on the right side of its subplot
        axis_config = {
            "title": title,
            "overlaying": primary_yaxis,
            "side": "right",
            "anchor": x_for_y[primary_yaxis],
            # Only show ticks on the rightmost secondary axis
            "showticklabels": is_rightmost,
            # Link non-rightmost axes to the rightmost for consistent scaling
            "matches": None if is_rightmost else rightmost_secondary_y,
            # Reserve margin space for tick labels and title so the legend
            # placed at x>=1 can't clip them.
            "automargin": True,
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

    base_n = len(base.data)
    sec_n = len(secondary.data)
    _ensure_legend_visibility(
        combined,
        [base, secondary],
        [slice(0, base_n), slice(base_n, base_n + sec_n)],
        cross_source_dedup=False,
    )
    _fix_animation_axis_ranges(combined)
    _set_default_secondary_y_layout(combined)
    return combined


def _set_default_secondary_y_layout(fig: go.Figure) -> None:
    """Anchor the legend to the figure container so it doesn't fight the
    secondary y-axis for paper-coordinate space.

    With ``xref="container"`` the legend's right edge sits at the figure's
    right edge regardless of plot width.  Combined with ``automargin=True``
    on the secondary y-axes (set in ``add_secondary_y``), Plotly reserves
    space for the axis title between the plot and the legend, so the two
    do not overlap.  Only fields the user has not already set are touched,
    so explicit ``update_layout(legend=...)`` on the source figures wins.
    """
    legend_defaults: dict[str, Any] = {}
    legend = fig.layout.legend
    if legend.x is None:
        # Container-relative x so the legend sits at the figure's right edge
        # rather than fighting the secondary y-axis title for paper-coord space.
        legend_defaults["x"] = 1.0
        legend_defaults["xanchor"] = "right"
        legend_defaults["xref"] = "container"
    if legend.y is None:
        # Paper-relative y so the legend top aligns with the plot top (below
        # the figure title) — same vertical position Plotly uses by default.
        legend_defaults["y"] = 1.0
        legend_defaults["yanchor"] = "top"
    if legend_defaults:
        fig.update_layout(legend=legend_defaults)


def _merge_secondary_y_frames(
    base: go.Figure,
    secondary: go.Figure,
    y_mapping: dict[str, str],
) -> list[go.Frame]:
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
                layout=base_frame.layout,
            )
        )

    return merged_frames


def _get_figure_title(fig: go.Figure) -> str:
    """Extract a display title from a figure for use as a subplot title.

    Checks, in order: the figure's title, then the y-axis title.

    Args:
        fig: A Plotly figure.

    Returns:
        A title string, or empty string if nothing is set.
    """
    try:
        title = fig.layout.title.text
        if isinstance(title, str) and title:
            return title
    except AttributeError:
        pass
    return _get_yaxis_title(fig)


def subplots(*figs: go.Figure, cols: int = 1) -> go.Figure:
    """Arrange multiple figures into a subplot grid.

    Creates a new figure with each input figure placed in its own cell.
    Figures may contain internal subplots (facets) — their axes are remapped
    to fit within the grid cell.  Subplot titles are derived from each
    figure's title or y-axis label.

    Args:
        *figs: One or more Plotly figures to arrange.
        cols: Number of columns in the grid. Rows are computed automatically.

    Returns:
        A new figure with subplot grid.

    Raises:
        ValueError: If no figures are provided, cols < 1, or a figure has
            animation frames.

    Example:
        >>> import numpy as np
        >>> import xarray as xr
        >>> from xarray_plotly import xpx, subplots
        >>>
        >>> temp = xr.DataArray([20, 22, 25], dims=["time"], name="Temperature")
        >>> rain = xr.DataArray([0, 5, 12], dims=["time"], name="Rainfall")
        >>> fig1 = xpx(temp).line()
        >>> fig2 = xpx(rain).bar()
        >>> grid = subplots(fig1, fig2, cols=2)
    """
    import math

    import plotly.graph_objects as go

    if not figs:
        raise ValueError("At least one figure is required.")
    if cols < 1:
        raise ValueError(f"cols must be >= 1, got {cols}.")

    for i, fig in enumerate(figs):
        if fig.frames:
            raise ValueError(
                f"Figure at position {i} has animation frames. "
                "Animated figures are not supported in subplots()."
            )

    rows = math.ceil(len(figs) / cols)
    combined = go.Figure()

    # Grid spacing
    h_gap = 0.05
    v_gap = 0.08
    cell_w = (1.0 - h_gap * (cols - 1)) / cols
    cell_h = (1.0 - v_gap * (rows - 1)) / rows

    next_x_num = 1
    next_y_num = 1

    for i, fig in enumerate(figs):
        row = i // cols  # 0-indexed, top to bottom
        col = i % cols

        # Cell boundaries (clamped to [0, 1])
        cell_x0 = max(0.0, col * (cell_w + h_gap))
        cell_x1 = min(1.0, cell_x0 + cell_w)
        cell_y1 = min(1.0, 1.0 - row * (cell_h + v_gap))  # top-down
        cell_y0 = max(0.0, cell_y1 - cell_h)

        # Build axis remapping: old axis ref → new axis ref
        axis_map, next_x_num, next_y_num = _remap_figure_axes(
            fig, combined, next_x_num, next_y_num, cell_x0, cell_x1, cell_y0, cell_y1
        )

        # Add traces with remapped axis refs
        for trace in fig.data:
            tc = copy.deepcopy(trace)
            old_x = getattr(tc, "xaxis", None) or "x"
            old_y = getattr(tc, "yaxis", None) or "y"
            tc.xaxis = axis_map[old_x]["new_x"]
            tc.yaxis = axis_map[old_y]["new_y"]
            combined.add_trace(tc)

        # Add subplot title as annotation
        title = _get_figure_title(fig)
        if title:
            combined.add_annotation(
                text=f"<b>{title}</b>",
                x=(cell_x0 + cell_x1) / 2,
                y=cell_y1,
                xref="paper",
                yref="paper",
                xanchor="center",
                yanchor="bottom",
                showarrow=False,
                font={"size": 14},
            )

    return combined


# Axis properties safe to copy between figures (display-only, not structural).
_AXIS_PROPS_TO_COPY = (
    "title",
    "type",
    "tickformat",
    "ticksuffix",
    "tickprefix",
    "dtick",
    "tick0",
    "nticks",
    "showgrid",
    "gridcolor",
    "gridwidth",
    "autorange",
    "range",
    "zeroline",
    "zerolinecolor",
    "zerolinewidth",
    "showticklabels",
)


def _axis_layout_key(ref: str) -> str:
    """Convert axis reference to layout property name.

    ``"x"`` → ``"xaxis"``, ``"x2"`` → ``"xaxis2"``,
    ``"y"`` → ``"yaxis"``, ``"y3"`` → ``"yaxis3"``.
    """
    if ref in ("x", "y"):
        return f"{ref}axis"
    prefix = ref[0]  # "x" or "y"
    num = ref[1:]
    return f"{prefix}axis{num}"


def _new_axis_ref(prefix: str, num: int) -> str:
    """Build an axis reference string. ``_new_axis_ref("x", 1)`` → ``"x"``, ``("x", 3)`` → ``"x3"``."""
    return prefix if num == 1 else f"{prefix}{num}"


def _remap_figure_axes(
    fig: go.Figure,
    combined: go.Figure,
    next_x_num: int,
    next_y_num: int,
    cell_x0: float,
    cell_x1: float,
    cell_y0: float,
    cell_y1: float,
) -> tuple[dict[str, dict[str, str]], int, int]:
    """Remap a figure's axes into a grid cell, adding axis configs to the combined layout.

    Args:
        fig: Source figure.
        combined: Target combined figure (mutated — axis configs added to layout).
        next_x_num: Next available x-axis number.
        next_y_num: Next available y-axis number.
        cell_x0, cell_x1: Horizontal cell bounds in paper coordinates.
        cell_y0, cell_y1: Vertical cell bounds in paper coordinates.

    Returns:
        Tuple of (axis_map, next_x_num, next_y_num).
        axis_map maps old axis refs to ``{"new_x": ...}`` or ``{"new_y": ...}``.
    """
    cell_w = cell_x1 - cell_x0
    cell_h = cell_y1 - cell_y0
    src_layout = fig.layout.to_plotly_json()

    x_remap: dict[str, str] = {}
    y_remap: dict[str, str] = {}

    # Get all unique axis refs
    x_refs: set[str] = set()
    y_refs: set[str] = set()
    for trace in fig.data:
        x_refs.add(getattr(trace, "xaxis", None) or "x")
        y_refs.add(getattr(trace, "yaxis", None) or "y")

    # Remap x-axes
    for old_xref in sorted(x_refs, key=lambda r: int(r[1:]) if len(r) > 1 else 1):
        new_xref = _new_axis_ref("x", next_x_num)
        x_remap[old_xref] = new_xref

        src_config = src_layout.get(_axis_layout_key(old_xref), {})
        src_domain = src_config.get("domain", [0.0, 1.0])
        new_domain = [
            max(0.0, cell_x0 + src_domain[0] * cell_w),
            min(1.0, cell_x0 + src_domain[1] * cell_w),
        ]

        new_config: dict[str, Any] = {"domain": new_domain}
        for prop in _AXIS_PROPS_TO_COPY:
            if prop in src_config:
                new_config[prop] = src_config[prop]

        combined.layout[_axis_layout_key(new_xref)] = new_config
        next_x_num += 1

    # Remap y-axes
    for old_yref in sorted(y_refs, key=lambda r: int(r[1:]) if len(r) > 1 else 1):
        new_yref = _new_axis_ref("y", next_y_num)
        y_remap[old_yref] = new_yref

        src_config = src_layout.get(_axis_layout_key(old_yref), {})
        src_domain = src_config.get("domain", [0.0, 1.0])
        new_domain = [
            max(0.0, cell_y0 + src_domain[0] * cell_h),
            min(1.0, cell_y0 + src_domain[1] * cell_h),
        ]

        new_config = {"domain": new_domain}
        for prop in _AXIS_PROPS_TO_COPY:
            if prop in src_config:
                new_config[prop] = src_config[prop]

        combined.layout[_axis_layout_key(new_yref)] = new_config
        next_y_num += 1

    # Set anchors between paired axes
    for trace in fig.data:
        old_x = getattr(trace, "xaxis", None) or "x"
        old_y = getattr(trace, "yaxis", None) or "y"
        combined.layout[_axis_layout_key(x_remap[old_x])]["anchor"] = y_remap[old_y]
        combined.layout[_axis_layout_key(y_remap[old_y])]["anchor"] = x_remap[old_x]

    # Propagate matches relationships
    for old_ref, new_ref in x_remap.items():
        src_config = src_layout.get(_axis_layout_key(old_ref), {})
        if "matches" in src_config and src_config["matches"] in x_remap:
            combined.layout[_axis_layout_key(new_ref)]["matches"] = x_remap[src_config["matches"]]

    for old_ref, new_ref in y_remap.items():
        src_config = src_layout.get(_axis_layout_key(old_ref), {})
        if "matches" in src_config and src_config["matches"] in y_remap:
            combined.layout[_axis_layout_key(new_ref)]["matches"] = y_remap[src_config["matches"]]

    # Build combined return mapping
    result: dict[str, dict[str, str]] = {}
    for old_x, new_x in x_remap.items():
        result[old_x] = {"new_x": new_x}
    for old_y, new_y in y_remap.items():
        result[old_y] = {"new_y": new_y}

    return result, next_x_num, next_y_num


def update_traces(
    fig: go.Figure, selector: dict[str, Any] | None = None, **kwargs: Any
) -> go.Figure:
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
