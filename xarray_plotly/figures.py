"""
Helper functions for combining and manipulating Plotly figures.
"""

from __future__ import annotations

import copy
import re
from collections import defaultdict
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

    import numpy as np
    import numpy.typing as npt
    import plotly.graph_objects as go

    from xarray_plotly.common import FacetTitlesMode


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


def _ensure_legend_visibility(
    combined: go.Figure,
    source_figs: list[go.Figure],
    trace_slices: list[slice],
) -> None:
    """Fix legend visibility on a combined figure.

    Handles three problems that arise when combining Plotly Express figures:

    1. **Unnamed traces** — PX sets ``name=""`` on single-trace (no color)
       figures.  We derive a name from each source figure's y-axis title.
    2. **Hidden named traces** — PX sets ``showlegend=False`` on single-trace
       figures.  We ensure at least one trace per ``legendgroup`` (or each
       ungrouped named trace) has ``showlegend=True``.
    3. **Duplicate legend entries** — when two source figures share the same
       ``legendgroup`` names, we deduplicate so only the first trace per
       group shows in the legend.

    Args:
        combined: The combined Plotly figure (mutated in place).
        source_figs: The original source figures, in trace order.
        trace_slices: Slices into ``combined.data`` for each source figure.
    """
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
    grouped: dict[str, list[Any]] = defaultdict(list)
    ungrouped: list[Any] = []

    for trace in combined.data:
        lg = getattr(trace, "legendgroup", None) or ""
        if lg:
            grouped[lg].append(trace)
        else:
            ungrouped.append(trace)

    for traces in grouped.values():
        has_visible = False
        for t in traces:
            if has_visible:
                # Deduplicate: only first keeps showlegend
                t.showlegend = False
            elif getattr(t, "name", None):
                t.showlegend = True
                has_visible = True

    # Ungrouped traces with a name should show in the legend
    for trace in ungrouped:
        if getattr(trace, "name", None):
            trace.showlegend = True

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


def _numeric_values(vals: Any) -> npt.NDArray[np.float64] | None:
    """Convert trace data to a 1-D float array, or None if not numeric.

    Datetime/timedelta and categorical (string) data return None — those
    axes are left on autorange.
    """
    import numpy as np

    if vals is None:
        return None
    arr = np.atleast_1d(np.asarray(vals))
    if np.issubdtype(arr.dtype, np.datetime64) or np.issubdtype(arr.dtype, np.timedelta64):
        return None
    try:
        return arr.astype(float)
    except (ValueError, TypeError):
        return None  # Non-numeric (categorical)


def _collect_axis_extents(traces: Any, stacked: bool) -> dict[tuple[str, str], list[float]]:
    """Collect candidate axis values for one trace collection.

    A "trace collection" is either ``fig.data`` or a single frame's data —
    stacking only happens between bars shown at the same time, so each
    collection must be aggregated independently.

    Args:
        traces: Iterable of traces (from ``fig.data`` or ``frame.data``).
        stacked: Whether the layout barmode stacks bars. When True, bar
            values on the value axis contribute per-category positive and
            negative sums instead of raw segment values.

    Returns:
        Mapping of ``(axis_letter, axis_ref)`` (e.g. ``("y", "y2")``) to the
        list of candidate values on that axis.
    """
    import numpy as np

    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    # (letter, axis_ref) -> category value -> [positive_sum, negative_sum]
    stack_sums: dict[tuple[str, str], dict[Any, list[float]]] = defaultdict(
        lambda: defaultdict(lambda: [0.0, 0.0])
    )

    for trace in traces:
        is_bar = getattr(trace, "type", None) == "bar"
        value_letter = "x" if (getattr(trace, "orientation", None) or "v") == "h" else "y"

        for letter in ("x", "y"):
            ref = getattr(trace, f"{letter}axis", None) or letter
            arr = _numeric_values(getattr(trace, letter, None))
            if arr is None:
                continue
            if is_bar and letter == value_letter:
                # Bars grow from a zero baseline, so 0 is part of the extent
                values[(letter, ref)].append(0.0)
                categories = getattr(trace, "y" if letter == "x" else "x", None)
                if stacked and categories is not None:
                    sums = stack_sums[(letter, ref)]
                    cat_list = np.atleast_1d(np.asarray(categories, dtype=object)).tolist()
                    for cat, val in zip(cat_list, arr.tolist(), strict=False):
                        if np.isfinite(val):
                            sums[cat][0 if val >= 0 else 1] += val
                    continue
            finite = arr[np.isfinite(arr)]
            if len(finite):
                values[(letter, ref)].extend(finite.tolist())

    # Stacked sums are the candidate extremes for the value axis
    for key, groups in stack_sums.items():
        for pos_sum, neg_sum in groups.values():
            values[key].extend((pos_sum, neg_sum))

    return values


def _fix_animation_axis_ranges(fig: go.Figure) -> None:
    """Pin axis ranges where animation frames exceed the initial view.

    Plotly.js computes autorange from ``fig.data`` only and does not
    recalculate during animation.  When a later frame has data outside the
    initial extent (e.g. population of Brazil vs China), values go
    off-screen.  This function computes the global min/max for each axis
    across all frames and sets an explicit range on the layout — but only
    for axes where that is actually necessary.

    Axes whose initial autorange already covers every frame are left on
    live autorange, so plotly's native padding is preserved and downstream
    changes (like switching the axis to ``type='category'``) keep working.
    Axes with a pre-set explicit range, non-linear axes (log, category,
    date), and non-numeric data are never touched.

    For ``barmode='stack'`` / ``'relative'``, bar extents are computed from
    per-category stacked sums rather than individual segment values, so
    tall stacks are not clipped.

    Args:
        fig: A Plotly figure with animation frames (mutated in place).
    """
    if not fig.frames:
        return

    stacked = fig.layout.barmode in ("stack", "relative")
    base_extents = _collect_axis_extents(fig.data, stacked)
    frame_extents = [_collect_axis_extents(frame.data, stacked) for frame in fig.frames]

    all_keys = set(base_extents) | {key for fe in frame_extents for key in fe}
    for key in sorted(all_keys):
        _letter, ref = key
        axis = fig.layout[_axis_layout_key(ref)]
        # Respect explicit ranges; log/date/category range coordinates are
        # not plain data values, so a computed range would corrupt the view.
        if axis.range is not None or axis.type in ("log", "date", "category", "multicategory"):
            continue

        base_vals = base_extents.get(key, [])
        all_vals = base_vals + [v for fe in frame_extents for v in fe.get(key, [])]
        if not all_vals:
            continue
        lo, hi = min(all_vals), max(all_vals)

        # Pin only when some frame exceeds the initial (fig.data) extent —
        # otherwise the autorange computed at first render stays valid for
        # the whole animation.
        if base_vals and min(base_vals) <= lo and max(base_vals) >= hi:
            continue

        pad = (hi - lo) * 0.05 or 1  # 5% padding
        axis.range = [lo - pad, hi + pad]


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
    )
    _fix_animation_axis_ranges(combined)
    return combined


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


# Matches an identifier-style PX facet prefix like "country=" at the start of
# annotation text.  Defensive: ignores annotations a user added themselves
# whose text doesn't look like a dim assignment.
_FACET_TITLE_PREFIX_RE = re.compile(r"^[A-Za-z_]\w*=")


def simplify_facet_titles(
    fig: go.Figure,
    mode: FacetTitlesMode = "value",
) -> go.Figure:
    """Strip the ``<dim>=`` prefix from Plotly Express facet subplot titles.

    PX renders faceted subplot titles as annotations like ``"country=Brazil"``.
    With ``mode="value"`` (default), the prefix is stripped to just the value
    (``"Brazil"``).  With ``mode="default"``, the figure is returned unchanged.

    Only annotations whose text matches a Python-identifier prefix followed by
    ``=`` are touched, so user-added annotations are left alone.

    Args:
        fig: A Plotly figure (mutated in place).
        mode: ``"value"`` to strip the prefix, ``"default"`` to keep PX's format.

    Returns:
        The (possibly mutated) figure, for chaining.

    Raises:
        ValueError: If ``mode`` is not ``"value"`` or ``"default"``.

    Example:
        >>> from xarray_plotly import xpx, simplify_facet_titles
        >>> fig = xpx(da).line(facet_col="country")
        >>> simplify_facet_titles(fig)  # "country=Brazil" -> "Brazil"
    """
    if mode == "default":
        return fig
    if mode != "value":
        raise ValueError(f"facet_titles must be 'value' or 'full', got {mode!r}")

    for ann in fig.layout.annotations or ():
        text = ann.text
        if text and _FACET_TITLE_PREFIX_RE.match(text):
            ann.text = text.split("=", 1)[1]
    return fig


# Matches cartesian axis layout keys like "xaxis", "xaxis2", "yaxis12".
_AXIS_KEY_RE = re.compile(r"([xy])axis\d*$")

# Annotation specs identical to plotly's built-in shared subplot titles,
# i.e. what `make_subplots(x_title=..., y_title=...)` produces.  Kept in
# sync with plotly via a test against make_subplots output.
_SHARED_LABEL_SPECS: dict[str, dict[str, Any]] = {
    "x": {
        "x": 0.5,
        "y": 0,
        "xref": "paper",
        "yref": "paper",
        "xanchor": "center",
        "yanchor": "top",
        "yshift": -30,
        "showarrow": False,
        "font": {"size": 16},
    },
    "y": {
        "x": 0,
        "y": 0.5,
        "xref": "paper",
        "yref": "paper",
        "xanchor": "right",
        "yanchor": "middle",
        "xshift": -40,
        "textangle": -90,
        "showarrow": False,
        "font": {"size": 16},
    },
}


def share_axis_labels(fig: go.Figure) -> go.Figure:
    """Replace repeated facet axis titles with a single shared label per axis.

    Plotly Express repeats the x-axis title under every facet column and the
    y-axis title beside every facet row.  This helper removes the repeated
    titles and adds one centered label per axis instead, styled exactly like
    plotly's built-in shared titles (``make_subplots(x_title=..., y_title=...)``),
    which Plotly Express does not expose for faceted figures.

    Titles are only collapsed when they are repeated and identical, so
    figures without facets, figures combined from differently-labeled
    subplots, and secondary-y figures are returned unchanged.

    Args:
        fig: A Plotly figure (mutated in place).

    Returns:
        The (possibly mutated) figure, for chaining.

    Example:
        >>> import plotly.express as px
        >>> from xarray_plotly import share_axis_labels
        >>> fig = px.line(df, x="year", y="gdp", facet_col="country", facet_row="metric")
        >>> share_axis_labels(fig)  # one "year" below, one "gdp" at the left
    """
    axes_by_letter: dict[str, list[Any]] = {"x": [], "y": []}
    for key in fig.layout:
        match = _AXIS_KEY_RE.match(key)
        # Overlaying axes (secondary y) share their domain with the axis
        # they overlay; their titles are independent, not facet repetition.
        if match and not fig.layout[key].overlaying:
            axes_by_letter[match.group(1)].append(fig.layout[key])

    for letter, axes in axes_by_letter.items():
        titles = [axis.title.text for axis in axes if axis.title.text]
        # Only collapse titles that are actually repeated and identical
        if len(titles) < 2 or len(set(titles)) != 1:
            continue
        for axis in axes:
            axis.title.text = None
        fig.add_annotation(text=titles[0], **_SHARED_LABEL_SPECS[letter])
    return fig
