"""
Configuration for xarray_plotly.

This module provides a global configuration system similar to xarray and pandas,
allowing users to customize label extraction and slot assignment behavior.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Generator


# Default slot orders per plot type
DEFAULT_SLOT_ORDERS: dict[str, tuple[str, ...]] = {
    "line": (
        "x",
        "color",
        "line_dash",
        "symbol",
        "facet_col",
        "facet_row",
        "animation_frame",
    ),
    "bar": ("x", "color", "pattern_shape", "facet_col", "facet_row", "animation_frame"),
    "area": (
        "x",
        "color",
        "pattern_shape",
        "facet_col",
        "facet_row",
        "animation_frame",
    ),
    "scatter": (
        "x",
        "color",
        "symbol",
        "facet_col",
        "facet_row",
        "animation_frame",
    ),
    "imshow": ("y", "x", "facet_col", "animation_frame"),
    "box": ("x", "color", "facet_col", "facet_row", "animation_frame"),
}


@dataclass
class Options:
    """
    Configuration options for xarray_plotly.

    Attributes
    ----------
    label_use_long_name : bool
        Use `long_name` attribute for labels. Default True.
    label_use_standard_name : bool
        Fall back to `standard_name` if `long_name` not available. Default True.
    label_include_units : bool
        Append units to labels. Default True.
    label_unit_format : str
        Format string for units. Use `{units}` as placeholder. Default "[{units}]".
    slot_orders : dict
        Slot orders per plot type. Keys are plot types, values are tuples of slot names.
    """

    label_use_long_name: bool = True
    label_use_standard_name: bool = True
    label_include_units: bool = True
    label_unit_format: str = "[{units}]"
    slot_orders: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: dict(DEFAULT_SLOT_ORDERS)
    )

    def to_dict(self) -> dict[str, Any]:
        """Return options as a dictionary."""
        return {
            "label_use_long_name": self.label_use_long_name,
            "label_use_standard_name": self.label_use_standard_name,
            "label_include_units": self.label_include_units,
            "label_unit_format": self.label_unit_format,
            "slot_orders": self.slot_orders,
        }


# Global options instance
_options = Options()


def get_options() -> dict[str, Any]:
    """
    Get the current xarray_plotly options.

    Returns
    -------
    dict
        Dictionary of current option values.

    Examples
    --------
    >>> from xarray_plotly import config
    >>> config.get_options()
    {'label_use_long_name': True, 'label_include_units': True, ...}
    """
    return _options.to_dict()


@contextmanager
def set_options(
    *,
    label_use_long_name: bool | None = None,
    label_use_standard_name: bool | None = None,
    label_include_units: bool | None = None,
    label_unit_format: str | None = None,
    slot_orders: dict[str, tuple[str, ...]] | None = None,
) -> Generator[None, None, None]:
    """
    Set xarray_plotly options globally or as a context manager.

    Parameters
    ----------
    label_use_long_name : bool, optional
        Use `long_name` attribute for labels.
    label_use_standard_name : bool, optional
        Fall back to `standard_name` if `long_name` not available.
    label_include_units : bool, optional
        Append units to labels.
    label_unit_format : str, optional
        Format string for units. Use `{units}` as placeholder.
    slot_orders : dict, optional
        Slot orders per plot type.

    Yields
    ------
    None
        When used as a context manager, yields nothing.

    Examples
    --------
    Set globally:

    >>> from xarray_plotly import config
    >>> config.set_options(label_include_units=False)

    Use as context manager:

    >>> with config.set_options(label_include_units=False):
    ...     fig = xpx(da).line()  # No units in labels
    >>> # Units are back after the context
    """
    # Store old values
    old_values = {
        "label_use_long_name": _options.label_use_long_name,
        "label_use_standard_name": _options.label_use_standard_name,
        "label_include_units": _options.label_include_units,
        "label_unit_format": _options.label_unit_format,
        "slot_orders": dict(_options.slot_orders),
    }

    # Apply new values (modify in place to keep reference)
    if label_use_long_name is not None:
        _options.label_use_long_name = label_use_long_name
    if label_use_standard_name is not None:
        _options.label_use_standard_name = label_use_standard_name
    if label_include_units is not None:
        _options.label_include_units = label_include_units
    if label_unit_format is not None:
        _options.label_unit_format = label_unit_format
    if slot_orders is not None:
        _options.slot_orders = dict(slot_orders)

    try:
        yield
    finally:
        # Restore old values (modify in place)
        _options.label_use_long_name = old_values["label_use_long_name"]
        _options.label_use_standard_name = old_values["label_use_standard_name"]
        _options.label_include_units = old_values["label_include_units"]
        _options.label_unit_format = old_values["label_unit_format"]
        _options.slot_orders = old_values["slot_orders"]


def notebook(renderer: str = "notebook") -> None:
    """
    Configure Plotly for Jupyter notebook rendering.

    Parameters
    ----------
    renderer : str, optional
        The Plotly renderer to use. Default is "notebook".
        Other options include "jupyterlab", "colab", "kaggle", etc.

    Examples
    --------
    >>> from xarray_plotly import config
    >>> config.notebook()  # Configure for Jupyter notebooks
    """
    import plotly.io as pio

    pio.renderers.default = renderer
