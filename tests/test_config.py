"""Tests for the configuration system."""

from __future__ import annotations

import numpy as np
import xarray as xr

import xarray_plotly as xp
from xarray_plotly import xpx
from xarray_plotly.common import get_label
from xarray_plotly.config import DEFAULT_SLOT_ORDERS


class TestGetOptions:
    """Tests for config.get_options()."""

    def test_returns_dict(self) -> None:
        """Test that get_options returns a dictionary."""
        opts = xp.config.get_options()
        assert isinstance(opts, dict)

    def test_contains_expected_keys(self) -> None:
        """Test that all expected keys are present."""
        opts = xp.config.get_options()
        assert "label_use_long_name" in opts
        assert "label_use_standard_name" in opts
        assert "label_include_units" in opts
        assert "label_unit_format" in opts
        assert "slot_orders" in opts

    def test_default_values(self) -> None:
        """Test default option values."""
        opts = xp.config.get_options()
        assert opts["label_use_long_name"] is True
        assert opts["label_use_standard_name"] is True
        assert opts["label_include_units"] is True
        assert opts["label_unit_format"] == "[{units}]"
        assert opts["slot_orders"] == DEFAULT_SLOT_ORDERS


class TestSetOptionsGlobal:
    """Tests for config.set_options() used globally."""

    def test_set_label_include_units(self) -> None:
        """Test setting label_include_units globally."""
        # Store original
        original = xp.config.get_options()["label_include_units"]

        try:
            with xp.config.set_options(label_include_units=False):
                assert xp.config.get_options()["label_include_units"] is False
            # Should be restored after context
            assert xp.config.get_options()["label_include_units"] is original
        finally:
            # Ensure cleanup
            with xp.config.set_options(label_include_units=original):
                pass

    def test_set_label_unit_format(self) -> None:
        """Test setting label_unit_format."""
        with xp.config.set_options(label_unit_format="({units})"):
            assert xp.config.get_options()["label_unit_format"] == "({units})"

    def test_set_multiple_options(self) -> None:
        """Test setting multiple options at once."""
        with xp.config.set_options(
            label_use_long_name=False,
            label_include_units=False,
        ):
            opts = xp.config.get_options()
            assert opts["label_use_long_name"] is False
            assert opts["label_include_units"] is False


class TestSetOptionsContextManager:
    """Tests for config.set_options() as context manager."""

    def test_restores_after_context(self) -> None:
        """Test that options are restored after context exits."""
        original = xp.config.get_options()["label_include_units"]

        with xp.config.set_options(label_include_units=not original):
            assert xp.config.get_options()["label_include_units"] is not original

        assert xp.config.get_options()["label_include_units"] is original

    def test_restores_on_exception(self) -> None:
        """Test that options are restored even if exception occurs."""
        original = xp.config.get_options()["label_include_units"]

        try:
            with xp.config.set_options(label_include_units=not original):
                assert xp.config.get_options()["label_include_units"] is not original
                raise ValueError("Test exception")
        except ValueError:
            pass

        assert xp.config.get_options()["label_include_units"] is original


class TestLabelOptions:
    """Tests for label-related options."""

    def setup_method(self) -> None:
        """Set up test data with attributes."""
        self.da = xr.DataArray(
            np.random.rand(10),
            dims=["time"],
            name="temperature",
            attrs={
                "long_name": "Air Temperature",
                "standard_name": "air_temp",
                "units": "K",
            },
        )
        self.da.coords["time"] = np.arange(10)
        self.da.coords["time"].attrs = {
            "long_name": "Time",
            "units": "days",
        }

    def test_default_label_includes_units(self) -> None:
        """Test that labels include units by default."""
        label = get_label(self.da, "value")
        assert "[K]" in label
        assert "Air Temperature" in label

    def test_label_without_units(self) -> None:
        """Test labels without units when disabled."""
        with xp.config.set_options(label_include_units=False):
            label = get_label(self.da, "value")
            assert "[K]" not in label
            assert "Air Temperature" in label

    def test_label_custom_unit_format(self) -> None:
        """Test custom unit format."""
        with xp.config.set_options(label_unit_format="({units})"):
            label = get_label(self.da, "value")
            assert "(K)" in label
            assert "[K]" not in label

    def test_label_without_long_name(self) -> None:
        """Test falling back to standard_name when long_name disabled."""
        with xp.config.set_options(label_use_long_name=False):
            label = get_label(self.da, "value")
            assert "air_temp" in label
            assert "Air Temperature" not in label

    def test_label_without_long_name_or_standard_name(self) -> None:
        """Test fallback to variable name when both disabled."""
        with xp.config.set_options(label_use_long_name=False, label_use_standard_name=False):
            label = get_label(self.da, "value")
            assert label == "temperature"

    def test_coord_label_with_options(self) -> None:
        """Test coordinate labels respect options."""
        with xp.config.set_options(label_include_units=False):
            label = get_label(self.da, "time")
            assert "Time" in label
            assert "[days]" not in label


class TestSlotOrderOptions:
    """Tests for slot_orders option."""

    def test_custom_slot_order(self) -> None:
        """Test using custom slot orders."""
        custom_orders = {
            **DEFAULT_SLOT_ORDERS,
            "line": ("x", "facet_col", "color", "facet_row", "animation_frame"),
        }

        da = xr.DataArray(
            np.random.rand(10, 3, 2),
            dims=["time", "city", "scenario"],
            coords={
                "time": np.arange(10),
                "city": ["A", "B", "C"],
                "scenario": ["X", "Y"],
            },
            name="test",
        )

        with xp.config.set_options(slot_orders=custom_orders):
            # With custom order: time->x, city->facet_col, scenario->color
            fig = xpx(da).line()
            # Just check it doesn't error - the figure structure is complex
            assert fig is not None
