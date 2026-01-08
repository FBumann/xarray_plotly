"""Tests for the slot assignment algorithm."""

from __future__ import annotations

import pytest

from xarray_plotly.common import SLOT_ORDERS, assign_slots


class TestAssignSlots:
    """Tests for the dimension-to-slot assignment algorithm."""

    def test_auto_assignment_line(self) -> None:
        """Test automatic positional assignment for line plots."""
        slots = assign_slots(["time", "city", "scenario"], "line")
        assert slots == {"x": "time", "color": "city", "line_dash": "scenario"}

    def test_auto_assignment_imshow(self) -> None:
        """Test automatic positional assignment for imshow."""
        slots = assign_slots(["lat", "lon"], "imshow")
        assert slots == {"y": "lat", "x": "lon"}

    def test_auto_assignment_scatter(self) -> None:
        """Test automatic positional assignment for scatter plots."""
        slots = assign_slots(["x_dim", "color_dim"], "scatter")
        assert slots == {"x": "x_dim", "color": "color_dim"}

    def test_auto_assignment_box(self) -> None:
        """Test automatic positional assignment for box plots."""
        slots = assign_slots(["category", "group"], "box")
        assert slots == {"x": "category", "color": "group"}

    def test_explicit_assignment(self) -> None:
        """Test explicit dimension-to-slot assignment."""
        slots = assign_slots(["time", "city", "scenario"], "line", x="city", color="time")
        assert slots["x"] == "city"
        assert slots["color"] == "time"
        assert slots["line_dash"] == "scenario"

    def test_skip_slot_with_none(self) -> None:
        """Test skipping a slot using None."""
        slots = assign_slots(["time", "city", "scenario"], "line", color=None)
        assert slots == {"x": "time", "line_dash": "city", "symbol": "scenario"}
        assert "color" not in slots

    def test_unassigned_dims_error(self) -> None:
        """Test that unassigned dimensions raise an error."""
        dims = list("abcdefgh")
        with pytest.raises(ValueError, match="Unassigned dimension"):
            assign_slots(dims, "line")

    def test_invalid_dimension_error(self) -> None:
        """Test that using a non-existent dimension raises an error."""
        with pytest.raises(ValueError, match="is not in the data dimensions"):
            assign_slots(["time", "city"], "line", x="nonexistent")

    def test_unknown_plot_type_error(self) -> None:
        """Test that unknown plot types raise an error."""
        with pytest.raises(ValueError, match="Unknown plot type"):
            assign_slots(["x", "y"], "unknown_plot_type")

    def test_all_explicit_assignment(self) -> None:
        """Test fully explicit assignment."""
        slots = assign_slots(["time", "city"], "line", x="city", color="time", facet_col=None)
        assert slots == {"x": "city", "color": "time"}

    def test_y_as_dimension_imshow(self) -> None:
        """Test assigning a dimension to y for imshow."""
        slots = assign_slots(["lat", "lon"], "imshow", y="lon", x="lat")
        assert slots == {"y": "lon", "x": "lat"}

    def test_allow_unassigned(self) -> None:
        """Test allowing unassigned dimensions with more dims than slots."""
        # box has 5 slots, but we have 6 dims - allow_unassigned lets this work
        slots = assign_slots(list("abcdef"), "box", allow_unassigned=True)
        assert slots == {"x": "a", "color": "b", "facet_col": "c", "facet_row": "d", "animation_frame": "e"}
        # 'f' is unassigned but no error is raised


class TestSlotOrders:
    """Tests for slot order configurations."""

    def test_all_plot_types_have_slot_orders(self) -> None:
        """Test that all plot types have defined slot orders."""
        assert "line" in SLOT_ORDERS
        assert "bar" in SLOT_ORDERS
        assert "area" in SLOT_ORDERS
        assert "scatter" in SLOT_ORDERS
        assert "box" in SLOT_ORDERS
        assert "imshow" in SLOT_ORDERS

    def test_line_slot_order(self) -> None:
        """Test line plot slot order."""
        assert SLOT_ORDERS["line"] == (
            "x",
            "color",
            "line_dash",
            "symbol",
            "facet_col",
            "facet_row",
            "animation_frame",
        )

    def test_scatter_slot_order(self) -> None:
        """Test scatter plot slot order."""
        assert SLOT_ORDERS["scatter"] == (
            "x",
            "color",
            "size",
            "symbol",
            "facet_col",
            "facet_row",
            "animation_frame",
        )

    def test_imshow_slot_order(self) -> None:
        """Test imshow slot order."""
        assert SLOT_ORDERS["imshow"] == (
            "y",
            "x",
            "facet_col",
            "animation_frame",
        )
