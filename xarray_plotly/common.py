"""Common utilities for dimension-to-slot assignment and data conversion."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from xarray_plotly.config import DEFAULT_SLOT_ORDERS, _options

if TYPE_CHECKING:
    from collections.abc import Hashable, Sequence

    import pandas as pd
    from xarray import DataArray

# Type definitions
PlotType = Literal["line", "bar", "area", "scatter", "imshow", "box", "pie"]
"""Available plot types."""

SlotName = Literal[
    "x",
    "y",
    "color",
    "line_dash",
    "symbol",
    "facet_col",
    "facet_row",
    "animation_frame",
    "pattern_shape",
    "names",
]
"""Available slot names for dimension assignment."""


class _AUTO:
    """Sentinel value for automatic slot assignment."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "auto"


auto = _AUTO()

SlotValue = _AUTO | str | None
"""Type alias for slot values: auto, explicit dimension name, or None (skip)."""

# Re-export for backward compatibility
SLOT_ORDERS = DEFAULT_SLOT_ORDERS
"""Slot orders per plot type.

For most plots, y-axis shows DataArray values (not a dimension slot).
For imshow, both y and x are dimensions (rows and columns of the heatmap).

Note:
    To customize slot orders, use `config.set_options(slot_orders=...)`.
"""


def assign_slots(
    dims: Sequence[Hashable],
    plot_type: PlotType,
    *,
    allow_unassigned: bool = False,
    **slot_kwargs: SlotValue,
) -> dict[str, Hashable]:
    """Assign dimensions to plot slots based on position.

    Positional assignment: dimensions fill slots in order.
    - Explicit assignments lock a dimension to a slot
    - None skips a slot
    - Remaining dims fill remaining slots by position
    - Error if dims left over after all slots filled (unless allow_unassigned=True)

    Args:
        dims: Dimension names from the DataArray.
        plot_type: Type of plot (line, bar, area, scatter, box, imshow).
        allow_unassigned: If True, allow dimensions to remain unassigned.
        **slot_kwargs: Explicit slot assignments. Use `auto` for positional,
            a dimension name for explicit, or `None` to skip.

    Returns:
        Mapping of slot names to dimension names.

    Raises:
        ValueError: If plot_type is unknown, dimension doesn't exist, or
            dimensions are left unassigned (unless allow_unassigned=True).

    Example:
        ```python
        assign_slots(["time", "city", "scenario"], "line")
        # {'x': 'time', 'color': 'city', 'line_dash': 'scenario'}

        assign_slots(["time", "city"], "line", color="time", x="city")
        # {'x': 'city', 'color': 'time'}

        assign_slots(["time", "city", "scenario"], "line", color=None)
        # {'x': 'time', 'line_dash': 'city', 'symbol': 'scenario'}
        ```
    """
    slot_orders = _options.slot_orders
    if plot_type not in slot_orders:
        msg = f"Unknown plot type: {plot_type!r}. Available types: {list(slot_orders.keys())}"
        raise ValueError(msg)

    slot_order = slot_orders[plot_type]
    dims_list = list(dims)

    slots: dict[str, Hashable] = {}
    used_dims: set[Hashable] = set()
    available_slots = list(slot_order)

    # Pass 1: Process explicit assignments (non-auto, non-None)
    for slot in slot_order:
        value = slot_kwargs.get(slot, auto)

        if value is None:
            # Skip this slot
            if slot in available_slots:
                available_slots.remove(slot)
        elif not isinstance(value, _AUTO):
            # Explicit assignment - can be a dimension name or "value" (DataArray values)
            if value == "value":
                slots[slot] = "value"
            elif value not in dims_list:
                msg = (
                    f"Dimension {value!r} assigned to slot {slot!r} "
                    f"is not in the data dimensions: {dims_list}"
                )
                raise ValueError(msg)
            else:
                slots[slot] = value
                used_dims.add(value)
            if slot in available_slots:
                available_slots.remove(slot)

    # Pass 2: Fill remaining slots with remaining dims (by position)
    remaining_dims = [d for d in dims_list if d not in used_dims]
    for slot, dim in zip(available_slots, remaining_dims, strict=False):
        slots[slot] = dim
        used_dims.add(dim)

    # Check for unassigned dimensions
    unassigned = [d for d in dims_list if d not in used_dims]
    if unassigned and not allow_unassigned:
        msg = (
            f"Unassigned dimension(s): {unassigned}. "
            "Reduce with .sel(), .isel(), or .mean() before plotting."
        )
        raise ValueError(msg)

    return slots


def get_value_col(darray: DataArray) -> str:
    """Get the column name for DataArray values."""
    return str(darray.name) if darray.name is not None else "value"


def to_dataframe(darray: DataArray) -> pd.DataFrame:
    """Convert a DataArray to a long-form DataFrame for Plotly Express."""
    if darray.name is None:
        darray = darray.rename("value")
    df: pd.DataFrame = darray.to_dataframe().reset_index()
    return df


def _get_label_from_attrs(attrs: dict[str, Any], fallback: str) -> str:
    """Extract a label from xarray attributes based on current config.

    Args:
        attrs: Attributes dictionary from DataArray or coordinate.
        fallback: Fallback label if no attributes match.

    Returns:
        The formatted label.
    """
    label = None

    if _options.label_use_long_name:
        label = attrs.get("long_name")

    if label is None and _options.label_use_standard_name:
        label = attrs.get("standard_name")

    if label is None:
        return fallback

    if _options.label_include_units:
        units = attrs.get("units")
        if units:
            return f"{label} {_options.label_unit_format.format(units=units)}"

    return str(label)


def get_label(darray: DataArray, name: Hashable) -> str:
    """Get a human-readable label for a dimension or the value column.

    Uses long_name/standard_name and units from attributes based on
    current configuration (see `config.set_options`).
    """
    # Check if it's asking for the value column label
    value_col = get_value_col(darray)
    if str(name) == value_col or name == "value":
        return _get_label_from_attrs(darray.attrs, value_col)

    # It's a dimension/coordinate
    if name in darray.coords:
        coord = darray.coords[name]
        return _get_label_from_attrs(coord.attrs, str(name))

    return str(name)


def build_labels(
    darray: DataArray,
    slots: dict[str, Hashable],
    value_col: str,
    *,
    include_value: bool = True,
) -> dict[str, str]:
    """Build a labels dict for Plotly Express from slot assignments.

    Args:
        darray: The source DataArray.
        slots: Slot assignments from assign_slots().
        value_col: The name of the value column in the DataFrame.
        include_value: Whether to include a label for the value column.

    Returns:
        Mapping of column names to human-readable labels.
    """
    labels: dict[str, str] = {}

    # Add labels for assigned dimensions
    for slot_value in slots.values():
        if slot_value and slot_value != "value":
            key = str(slot_value)
            if key not in labels:
                labels[key] = get_label(darray, slot_value)

    # Add label for value column
    if include_value and value_col not in labels:
        labels[value_col] = get_label(darray, "value")

    return labels
