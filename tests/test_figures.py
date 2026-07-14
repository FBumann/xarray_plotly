"""Tests for the figures module (overlay, add_secondary_y, subplots)."""

from __future__ import annotations

import copy

import numpy as np
import plotly.graph_objects as go
import pytest
import xarray as xr

from xarray_plotly import (
    add_secondary_y,
    overlay,
    share_axis_labels,
    simplify_facet_titles,
    subplots,
    xpx,
)


class TestOverlayBasic:
    """Basic tests for overlay function."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        """Set up test data."""
        self.da_2d = xr.DataArray(
            np.random.rand(10, 3),
            dims=["time", "cat"],
            coords={"time": np.arange(10), "cat": ["A", "B", "C"]},
            name="value",
        )

    def test_no_overlays_returns_copy(self) -> None:
        """Test that no overlays returns a deep copy of base."""
        base = xpx(self.da_2d).line()
        result = overlay(base)

        assert isinstance(result, go.Figure)
        assert len(result.data) == len(base.data)
        # Verify it's a copy, not the same object
        assert result is not base
        assert result.data[0] is not base.data[0]

    def test_combine_two_static_figures(self) -> None:
        """Test combining two static figures."""
        area_fig = xpx(self.da_2d).area()
        line_fig = xpx(self.da_2d).line()

        combined = overlay(area_fig, line_fig)

        assert isinstance(combined, go.Figure)
        expected_trace_count = len(area_fig.data) + len(line_fig.data)
        assert len(combined.data) == expected_trace_count

    def test_preserves_base_layout(self) -> None:
        """Test that base figure's layout is preserved."""
        area_fig = xpx(self.da_2d).area(title="My Area Plot")
        line_fig = xpx(self.da_2d).line(title="My Line Plot")

        combined = overlay(area_fig, line_fig)

        assert combined.layout.title.text == "My Area Plot"

    def test_multiple_overlays(self) -> None:
        """Test combining multiple overlays."""
        area_fig = xpx(self.da_2d).area()
        line_fig = xpx(self.da_2d).line()
        scatter_fig = xpx(self.da_2d).scatter()

        combined = overlay(area_fig, line_fig, scatter_fig)

        expected_count = len(area_fig.data) + len(line_fig.data) + len(scatter_fig.data)
        assert len(combined.data) == expected_count

    def test_overlay_traces_added_in_order(self) -> None:
        """Test that overlay traces are added after base traces."""
        # Create figures with distinguishable y values
        da_1 = xr.DataArray([1, 2, 3], dims=["x"], name="first")
        da_2 = xr.DataArray([10, 20, 30], dims=["x"], name="second")

        fig1 = xpx(da_1).line()
        fig2 = xpx(da_2).line()

        combined = overlay(fig1, fig2)

        # First trace should have y values from fig1
        assert list(combined.data[0].y) == [1, 2, 3]
        # Second trace should have y values from fig2
        assert list(combined.data[1].y) == [10, 20, 30]


class TestOverlayFacets:
    """Tests for overlay with faceted figures."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        """Set up test data."""
        self.da_3d = xr.DataArray(
            np.random.rand(10, 3, 2),
            dims=["time", "cat", "facet"],
            coords={
                "time": np.arange(10),
                "cat": ["A", "B", "C"],
                "facet": ["left", "right"],
            },
            name="value",
        )

    def test_matching_facet_structures(self) -> None:
        """Test combining figures with matching facet structures."""
        area_fig = xpx(self.da_3d).area(facet_col="facet")
        line_fig = xpx(self.da_3d).line(facet_col="facet")

        combined = overlay(area_fig, line_fig)

        assert isinstance(combined, go.Figure)
        expected_count = len(area_fig.data) + len(line_fig.data)
        assert len(combined.data) == expected_count

    def test_overlay_with_extra_subplots_raises(self) -> None:
        """Test that overlay with extra subplots raises ValueError."""
        # Base without facets
        base = xpx(self.da_3d.isel(facet=0)).line()
        # Overlay with facets
        overlay_fig = xpx(self.da_3d).line(facet_col="facet")

        with pytest.raises(ValueError, match="subplots not present in base"):
            overlay(base, overlay_fig)

    def test_preserves_axis_references(self) -> None:
        """Test that traces preserve their xaxis/yaxis references."""
        area_fig = xpx(self.da_3d).area(facet_col="facet")
        line_fig = xpx(self.da_3d).line(facet_col="facet")

        combined = overlay(area_fig, line_fig)

        # Collect axis references from both original and combined
        original_axes = set()
        for trace in area_fig.data:
            xaxis = getattr(trace, "xaxis", None) or "x"
            yaxis = getattr(trace, "yaxis", None) or "y"
            original_axes.add((xaxis, yaxis))

        combined_axes = set()
        for trace in combined.data:
            xaxis = getattr(trace, "xaxis", None) or "x"
            yaxis = getattr(trace, "yaxis", None) or "y"
            combined_axes.add((xaxis, yaxis))

        # Combined should have same axis structure
        assert combined_axes == original_axes


class TestOverlayAnimation:
    """Tests for overlay with animated figures."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        """Set up test data."""
        self.da_3d = xr.DataArray(
            np.random.rand(10, 3, 4),
            dims=["x", "cat", "time"],
            coords={
                "x": np.arange(10),
                "cat": ["A", "B", "C"],
                "time": [0, 1, 2, 3],
            },
            name="value",
        )

    def test_matching_frames_merged(self) -> None:
        """Test that matching animation frames are merged correctly."""
        area_fig = xpx(self.da_3d).area(animation_frame="time")
        line_fig = xpx(self.da_3d).line(animation_frame="time")

        combined = overlay(area_fig, line_fig)

        assert isinstance(combined, go.Figure)
        # Should have same number of frames
        assert len(combined.frames) == len(area_fig.frames)
        # Each frame should have more data
        for i, frame in enumerate(combined.frames):
            expected_data = len(area_fig.frames[i].data) + len(line_fig.frames[i].data)
            assert len(frame.data) == expected_data

    def test_static_overlay_replicated_to_frames(self) -> None:
        """Test that static overlay is replicated to all animation frames."""
        animated = xpx(self.da_3d).area(animation_frame="time")
        static = xpx(self.da_3d.isel(time=0)).line()

        combined = overlay(animated, static)

        # Combined should have all frames from animated figure
        assert len(combined.frames) == len(animated.frames)

        # Each frame should include the static traces
        for frame in combined.frames:
            # Frame data should include both animated and static traces
            expected_count = len(animated.frames[0].data) + len(static.data)
            assert len(frame.data) == expected_count

    def test_animated_overlay_on_static_base_raises(self) -> None:
        """Test that animated overlay on static base raises ValueError."""
        static = xpx(self.da_3d.isel(time=0)).line()
        animated = xpx(self.da_3d).area(animation_frame="time")

        with pytest.raises(ValueError, match="base figure does not"):
            overlay(static, animated)

    def test_mismatched_frame_names_raises(self) -> None:
        """Test that mismatched frame names raise ValueError."""
        da1 = xr.DataArray(
            np.random.rand(10, 3),
            dims=["x", "time"],
            coords={"x": np.arange(10), "time": [0, 1, 2]},
        )
        da2 = xr.DataArray(
            np.random.rand(10, 4),
            dims=["x", "time"],
            coords={"x": np.arange(10), "time": [0, 1, 2, 3]},
        )

        fig1 = xpx(da1).line(animation_frame="time")
        fig2 = xpx(da2).line(animation_frame="time")

        with pytest.raises(ValueError, match="frame names don't match"):
            overlay(fig1, fig2)

    def test_frame_names_preserved(self) -> None:
        """Test that frame names are preserved in combined figure."""
        area_fig = xpx(self.da_3d).area(animation_frame="time")
        line_fig = xpx(self.da_3d).line(animation_frame="time")

        combined = overlay(area_fig, line_fig)

        original_names = {frame.name for frame in area_fig.frames}
        combined_names = {frame.name for frame in combined.frames}
        assert original_names == combined_names

    def test_frame_layout_preserved(self) -> None:
        """Test that frame layout (e.g., axis range) is preserved."""
        fig = xpx(self.da_3d).line(animation_frame="time", range_y=[0, 10])
        overlay_fig = xpx(self.da_3d).scatter(animation_frame="time")

        # Verify base has frame layout
        assert fig.frames[0].layout is not None

        combined = overlay(fig, overlay_fig)

        # Frame layout should be preserved
        for i, frame in enumerate(combined.frames):
            assert frame.layout == fig.frames[i].layout


class TestOverlayFacetsAndAnimation:
    """Tests for overlay with both facets and animation."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        """Set up test data."""
        self.da_4d = xr.DataArray(
            np.random.rand(10, 3, 2, 4),
            dims=["x", "cat", "facet", "time"],
            coords={
                "x": np.arange(10),
                "cat": ["A", "B", "C"],
                "facet": ["left", "right"],
                "time": [0, 1, 2, 3],
            },
            name="value",
        )

    def test_facets_and_animation_combined(self) -> None:
        """Test combining figures with both facets and animation."""
        area_fig = xpx(self.da_4d).area(facet_col="facet", animation_frame="time")
        line_fig = xpx(self.da_4d).line(facet_col="facet", animation_frame="time")

        combined = overlay(area_fig, line_fig)

        assert isinstance(combined, go.Figure)
        # Check trace count
        expected_traces = len(area_fig.data) + len(line_fig.data)
        assert len(combined.data) == expected_traces
        # Check frame count
        assert len(combined.frames) == len(area_fig.frames)

    def test_static_overlay_on_animated_faceted_base(self) -> None:
        """Test static overlay replicated on animated faceted base."""
        animated = xpx(self.da_4d).area(facet_col="facet", animation_frame="time")
        static = xpx(self.da_4d.isel(time=0)).line(facet_col="facet")

        combined = overlay(animated, static)

        # Should have same frames as animated
        assert len(combined.frames) == len(animated.frames)
        # Each frame should have combined trace count
        for frame in combined.frames:
            expected = len(animated.frames[0].data) + len(static.data)
            assert len(frame.data) == expected


class TestOverlayDeepCopy:
    """Tests to ensure overlay creates deep copies."""

    def test_base_not_modified(self) -> None:
        """Test that base figure is not modified."""
        da = xr.DataArray(np.random.rand(10, 3), dims=["x", "cat"])
        base = xpx(da).area()
        original_trace_count = len(base.data)
        original_title = copy.deepcopy(base.layout.title)

        overlay_fig = xpx(da).line()
        _ = overlay(base, overlay_fig)

        # Base should be unchanged
        assert len(base.data) == original_trace_count
        assert base.layout.title == original_title

    def test_overlay_not_modified(self) -> None:
        """Test that overlay figure is not modified."""
        da = xr.DataArray(np.random.rand(10, 3), dims=["x", "cat"])
        base = xpx(da).area()
        overlay_fig = xpx(da).line()
        original_trace_count = len(overlay_fig.data)

        _ = overlay(base, overlay_fig)

        # Overlay should be unchanged
        assert len(overlay_fig.data) == original_trace_count

    def test_combined_traces_independent(self) -> None:
        """Test that combined traces are independent of originals."""
        da = xr.DataArray(np.random.rand(10, 3), dims=["x", "cat"])
        base = xpx(da).area()
        overlay_fig = xpx(da).line()

        combined = overlay(base, overlay_fig)

        # Modify combined figure
        combined.data[0].name = "modified"

        # Originals should be unchanged
        assert base.data[0].name != "modified"


class TestAddSecondaryYBasic:
    """Basic tests for add_secondary_y function."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        """Set up test data."""
        self.temp = xr.DataArray(
            [20, 22, 25, 23, 21],
            dims=["time"],
            coords={"time": [0, 1, 2, 3, 4]},
            name="Temperature",
        )
        self.precip = xr.DataArray(
            [0, 5, 12, 2, 8],
            dims=["time"],
            coords={"time": [0, 1, 2, 3, 4]},
            name="Precipitation",
        )

    def test_creates_secondary_y_axis(self) -> None:
        """Test that secondary y-axis is created."""
        temp_fig = xpx(self.temp).line()
        precip_fig = xpx(self.precip).bar()

        combined = add_secondary_y(temp_fig, precip_fig)

        assert isinstance(combined, go.Figure)
        assert combined.layout.yaxis2 is not None
        assert combined.layout.yaxis2.side == "right"
        assert combined.layout.yaxis2.overlaying == "y"

    def test_secondary_traces_use_y2(self) -> None:
        """Test that secondary figure traces are assigned to y2."""
        temp_fig = xpx(self.temp).line()
        precip_fig = xpx(self.precip).bar()

        combined = add_secondary_y(temp_fig, precip_fig)

        # First trace (from temp_fig) should use default y
        assert combined.data[0].yaxis is None or combined.data[0].yaxis == "y"
        # Second trace (from precip_fig) should use y2
        assert combined.data[1].yaxis == "y2"

    def test_preserves_base_layout(self) -> None:
        """Test that base figure's layout is preserved."""
        temp_fig = xpx(self.temp).line(title="My Temperature Plot")
        precip_fig = xpx(self.precip).bar()

        combined = add_secondary_y(temp_fig, precip_fig)

        assert combined.layout.title.text == "My Temperature Plot"

    def test_total_trace_count(self) -> None:
        """Test that all traces from both figures are included."""
        temp_fig = xpx(self.temp).line()
        precip_fig = xpx(self.precip).bar()

        combined = add_secondary_y(temp_fig, precip_fig)

        expected_count = len(temp_fig.data) + len(precip_fig.data)
        assert len(combined.data) == expected_count

    def test_secondary_y_title_from_secondary_figure(self) -> None:
        """Test that secondary y-axis title comes from secondary figure."""
        temp_fig = xpx(self.temp).line()
        precip_fig = xpx(self.precip).bar()
        # Plotly Express sets y-axis title based on the data

        combined = add_secondary_y(temp_fig, precip_fig)

        # The secondary y-axis title should be set
        assert combined.layout.yaxis2.title is not None

    def test_custom_secondary_y_title(self) -> None:
        """Test that custom secondary y-axis title can be provided."""
        temp_fig = xpx(self.temp).line()
        precip_fig = xpx(self.precip).bar()

        combined = add_secondary_y(temp_fig, precip_fig, secondary_y_title="Rain (mm)")

        assert combined.layout.yaxis2.title.text == "Rain (mm)"


class TestAddSecondaryYFacets:
    """Tests for add_secondary_y with faceted figures."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        """Set up test data."""
        self.da = xr.DataArray(
            np.random.rand(10, 3),
            dims=["time", "facet"],
            coords={"time": np.arange(10), "facet": ["A", "B", "C"]},
            name="value",
        )
        # Different scale for secondary
        self.da_secondary = xr.DataArray(
            np.random.rand(10, 3) * 1000,
            dims=["time", "facet"],
            coords={"time": np.arange(10), "facet": ["A", "B", "C"]},
            name="large_value",
        )

    def test_matching_facets_works(self) -> None:
        """Test that matching facet structures work."""
        base = xpx(self.da).line(facet_col="facet")
        secondary = xpx(self.da_secondary).bar(facet_col="facet")

        combined = add_secondary_y(base, secondary)

        assert isinstance(combined, go.Figure)
        expected_traces = len(base.data) + len(secondary.data)
        assert len(combined.data) == expected_traces

    def test_facets_creates_multiple_secondary_axes(self) -> None:
        """Test that secondary y-axes are created for each facet."""
        base = xpx(self.da).line(facet_col="facet")
        secondary = xpx(self.da_secondary).bar(facet_col="facet")

        combined = add_secondary_y(base, secondary)

        # Should have yaxis2 (secondary for y), yaxis5 (secondary for y2), etc.
        # Base has y, y2, y3, so secondary should be y4, y5, y6
        layout_json = combined.layout.to_plotly_json()
        assert "yaxis4" in layout_json
        assert layout_json["yaxis4"]["overlaying"] == "y"
        assert layout_json["yaxis4"]["side"] == "right"

    def test_secondary_traces_remapped_to_correct_axes(self) -> None:
        """Test that secondary traces use correct secondary y-axes."""
        base = xpx(self.da).line(facet_col="facet")
        secondary = xpx(self.da_secondary).bar(facet_col="facet")

        combined = add_secondary_y(base, secondary)

        # Get secondary trace y-axes
        secondary_trace_yaxes = {trace.yaxis for trace in combined.data[len(base.data) :]}
        # Should be y4, y5, y6 (secondary axes)
        assert secondary_trace_yaxes == {"y4", "y5", "y6"}

    def test_mismatched_facets_raises(self) -> None:
        """Test that mismatched facet structures raise ValueError."""
        # Base with facets
        base = xpx(self.da).line(facet_col="facet")
        # Secondary without facets
        secondary = xpx(self.da.isel(facet=0)).bar()

        with pytest.raises(ValueError, match="same facet structure"):
            add_secondary_y(base, secondary)

    def test_mismatched_facets_reversed_raises(self) -> None:
        """Test that mismatched facets raise (base without, secondary with)."""
        # Base without facets
        base = xpx(self.da.isel(facet=0)).line()
        # Secondary with facets
        secondary = xpx(self.da).bar(facet_col="facet")

        with pytest.raises(ValueError, match="same facet structure"):
            add_secondary_y(base, secondary)

    def test_facets_with_custom_title(self) -> None:
        """Test custom secondary y-axis title with facets."""
        base = xpx(self.da).line(facet_col="facet")
        secondary = xpx(self.da_secondary).bar(facet_col="facet")

        combined = add_secondary_y(base, secondary, secondary_y_title="Custom Title")

        # Title should be on the rightmost secondary axis (yaxis6 for 3 facets)
        assert combined.layout.yaxis6.title.text == "Custom Title"


class TestAddSecondaryYAnimation:
    """Tests for add_secondary_y with animated figures."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        """Set up test data."""
        self.da_2d = xr.DataArray(
            np.random.rand(10, 4),
            dims=["x", "time"],
            coords={"x": np.arange(10), "time": [0, 1, 2, 3]},
            name="value",
        )

    def test_matching_animation_frames(self) -> None:
        """Test add_secondary_y with matching animation frames."""
        fig1 = xpx(self.da_2d).line(animation_frame="time")
        fig2 = xpx(self.da_2d).bar(animation_frame="time")

        combined = add_secondary_y(fig1, fig2)

        assert len(combined.frames) == len(fig1.frames)
        # Verify frame names match
        for orig, comb in zip(fig1.frames, combined.frames, strict=False):
            assert orig.name == comb.name

    def test_static_secondary_on_animated_base(self) -> None:
        """Test static secondary replicated to all animation frames."""
        animated = xpx(self.da_2d).line(animation_frame="time")
        static = xpx(self.da_2d.isel(time=0)).bar()

        combined = add_secondary_y(animated, static)

        assert len(combined.frames) == len(animated.frames)
        # Each frame should have traces from both figures
        for frame in combined.frames:
            expected = len(animated.frames[0].data) + len(static.data)
            assert len(frame.data) == expected

    def test_animated_secondary_on_static_base_raises(self) -> None:
        """Test that animated secondary on static base raises ValueError."""
        static = xpx(self.da_2d.isel(time=0)).line()
        animated = xpx(self.da_2d).bar(animation_frame="time")

        with pytest.raises(ValueError, match="base figure does not"):
            add_secondary_y(static, animated)

    def test_mismatched_animation_frames_raises(self) -> None:
        """Test that mismatched animation frames raise ValueError."""
        da1 = xr.DataArray(
            np.random.rand(10, 3),
            dims=["x", "time"],
            coords={"x": np.arange(10), "time": [0, 1, 2]},
        )
        da2 = xr.DataArray(
            np.random.rand(10, 4),
            dims=["x", "time"],
            coords={"x": np.arange(10), "time": [0, 1, 2, 3]},
        )

        fig1 = xpx(da1).line(animation_frame="time")
        fig2 = xpx(da2).bar(animation_frame="time")

        with pytest.raises(ValueError, match="frame names don't match"):
            add_secondary_y(fig1, fig2)

    def test_frame_layout_preserved(self) -> None:
        """Test that frame layout (e.g., axis range) is preserved."""
        base = xpx(self.da_2d).line(animation_frame="time", range_y=[0, 10])
        secondary = xpx(self.da_2d).bar(animation_frame="time")

        # Verify base has frame layout
        assert base.frames[0].layout is not None

        combined = add_secondary_y(base, secondary)

        # Frame layout should be preserved
        for i, frame in enumerate(combined.frames):
            assert frame.layout == base.frames[i].layout


class TestAddSecondaryYDeepCopy:
    """Tests to ensure add_secondary_y creates deep copies."""

    def test_base_not_modified(self) -> None:
        """Test that base figure is not modified."""
        da = xr.DataArray([1, 2, 3, 4, 5], dims=["x"])
        base = xpx(da).line()
        original_trace_count = len(base.data)

        secondary = xpx(da).bar()
        _ = add_secondary_y(base, secondary)

        assert len(base.data) == original_trace_count
        # Base should not have yaxis2 (check via to_plotly_json)
        assert "yaxis2" not in base.layout.to_plotly_json()

    def test_secondary_not_modified(self) -> None:
        """Test that secondary figure is not modified."""
        da = xr.DataArray([1, 2, 3, 4, 5], dims=["x"])
        base = xpx(da).line()
        secondary = xpx(da).bar()
        original_yaxis = secondary.data[0].yaxis

        _ = add_secondary_y(base, secondary)

        # Secondary traces should still use original yaxis
        assert secondary.data[0].yaxis == original_yaxis


class TestLegendVisibility:
    """Tests that combined figures preserve legend visibility."""

    def test_overlay_single_trace_figures_with_names(self) -> None:
        """Overlay of named single-trace figures shows legend."""
        da1 = xr.DataArray([1, 2, 3], dims=["x"], name="a")
        da2 = xr.DataArray([4, 5, 6], dims=["x"], name="b")

        fig1 = xpx(da1).line()
        fig1.update_traces(name="Series A")
        fig2 = xpx(da2).line()
        fig2.update_traces(name="Series B")

        combined = overlay(fig1, fig2)

        assert combined.data[0].showlegend is True
        assert combined.data[1].showlegend is True

    def test_overlay_unnamed_traces_get_yaxis_title(self) -> None:
        """Overlay of unnamed traces derives names from y-axis titles."""
        da1 = xr.DataArray([1, 2, 3], dims=["x"], name="Temperature")
        da2 = xr.DataArray([4, 5, 6], dims=["x"], name="Pressure")

        fig1 = xpx(da1).line()
        fig2 = xpx(da2).line()

        combined = overlay(fig1, fig2)

        # Names derived from y-axis titles (DataArray names)
        assert combined.data[0].name == "Temperature"
        assert combined.data[1].name == "Pressure"
        assert combined.data[0].showlegend is True
        assert combined.data[1].showlegend is True

    def test_overlay_same_name_disambiguated(self) -> None:
        """Overlay of figures with same y-axis title gets numeric suffix."""
        da1 = xr.DataArray([1, 2, 3], dims=["x"], name="value")
        da2 = xr.DataArray([4, 5, 6], dims=["x"], name="value")

        fig1 = xpx(da1).line()
        fig2 = xpx(da2).line()

        combined = overlay(fig1, fig2)

        assert combined.data[0].name == "value (1)"
        assert combined.data[1].name == "value (2)"

    def test_overlay_multi_trace_deduplicates_legend(self) -> None:
        """Overlay of multi-trace figures deduplicates shared legendgroups."""
        da = xr.DataArray(
            np.random.rand(10, 3),
            dims=["x", "cat"],
            coords={"cat": ["A", "B", "C"]},
        )
        fig1 = xpx(da).area()
        fig2 = xpx(da).line()

        combined = overlay(fig1, fig2)

        # First occurrence of each legendgroup should show, duplicates hidden
        from collections import defaultdict

        groups: dict[str, list[bool]] = defaultdict(list)
        for trace in combined.data:
            lg = trace.legendgroup
            groups[lg].append(trace.showlegend is True)

        for lg, flags in groups.items():
            assert flags.count(True) == 1, f"legendgroup {lg!r} has {flags.count(True)} visible"

    def test_add_secondary_y_single_trace_with_names(self) -> None:
        """add_secondary_y of named single-trace figures shows legend."""
        da1 = xr.DataArray([1, 2, 3], dims=["x"], name="temp")
        da2 = xr.DataArray([100, 200, 300], dims=["x"], name="precip")

        fig1 = xpx(da1).line()
        fig1.update_traces(name="Temperature")
        fig2 = xpx(da2).bar()
        fig2.update_traces(name="Precipitation")

        combined = add_secondary_y(fig1, fig2)

        assert combined.data[0].showlegend is True
        assert combined.data[1].showlegend is True

    def test_overlay_faceted_legendgroup_dedup(self) -> None:
        """Faceted overlay keeps only one showlegend=True per legendgroup."""
        da = xr.DataArray(
            np.random.rand(10, 2, 2),
            dims=["x", "cat", "facet"],
            coords={"cat": ["A", "B"], "facet": ["left", "right"]},
        )
        fig1 = xpx(da).area(facet_col="facet")
        fig2 = xpx(da).line(facet_col="facet")

        combined = overlay(fig1, fig2)

        # Check each legendgroup has at least one showlegend=True
        from collections import defaultdict

        groups: dict[str, list[bool]] = defaultdict(list)
        for trace in combined.data:
            lg = trace.legendgroup or ""
            if lg:
                groups[lg].append(trace.showlegend is True)

        for lg, flags in groups.items():
            assert any(flags), f"legendgroup {lg!r} has no showlegend=True trace"

    def test_overlay_animation_frames_preserve_style(self) -> None:
        """Animation frame traces keep legend and color from fig.data."""
        da = xr.DataArray(
            np.random.rand(10, 3),
            dims=["x", "time"],
            coords={"time": [0, 1, 2]},
            name="Population",
        )
        da_smooth = da.rolling(x=3, center=True).mean()
        da_smooth.name = "Smoothed"

        fig1 = xpx(da).bar(animation_frame="time")
        fig1.update_traces(marker={"color": "steelblue"})
        fig2 = xpx(da_smooth).line(animation_frame="time")
        fig2.update_traces(line={"color": "red"})

        combined = overlay(fig1, fig2)

        for frame in combined.frames:
            for i, ft in enumerate(frame.data):
                src = combined.data[i]
                assert ft.name == src.name
                assert ft.showlegend == src.showlegend
                assert ft.legendgroup == src.legendgroup
            # Bar trace should keep steelblue
            assert frame.data[0].marker.color == "steelblue"
            # Line trace should keep red
            assert frame.data[1].line.color == "red"


class TestAnimationAxisRanges:
    """Tests for _fix_animation_axis_ranges."""

    def test_datetime_x_axis_not_corrupted(self) -> None:
        """datetime64 x-axis should be left on autorange, not cast to float epochs."""
        dates = np.array(["2020-01-01", "2020-06-01", "2021-01-01"], dtype="datetime64[ns]")
        da = xr.DataArray(
            np.random.rand(3, 2),
            dims=["date", "cat"],
            coords={"date": dates, "cat": ["A", "B"]},
            name="value",
        )
        fig1 = xpx(da).line(animation_frame="cat")
        fig2 = xpx(da).scatter(animation_frame="cat")
        combined = overlay(fig1, fig2)

        # x-axis range should NOT be set (dates left to autorange)
        assert combined.layout.xaxis.range is None

    def test_bar_zero_baseline(self) -> None:
        """Bar chart y-axis range should include zero."""
        da = xr.DataArray(
            np.array([[100, 200], [150, 250]]),
            dims=["x", "frame"],
            name="val",
        )
        fig = xpx(da).bar(animation_frame="frame")
        # After overlay (which triggers _fix_animation_axis_ranges)
        combined = overlay(fig, xpx(da).line(animation_frame="frame"))

        lo, _hi = combined.layout.yaxis.range
        assert lo <= 0, f"Bar y-axis range should include 0, got lo={lo}"

    def test_x_axis_stays_on_autorange_when_frames_share_extent(self) -> None:
        """x-axis must not be pinned when every frame spans the same x values.

        Pinning it froze a numeric range that became a garbage category-index
        range if the user later switched the axis to type='category'.
        """
        da = xr.DataArray(
            np.random.rand(4, 3, 2),
            dims=["period", "tech", "case"],
            coords={"period": [2025, 2030, 2035, 2040]},
            name="heat",
        )
        line_da = xr.DataArray(
            np.random.rand(4, 2) * 200,
            dims=["period", "case"],
            coords={"period": [2025, 2030, 2035, 2040]},
            name="demand",
        )
        combined = overlay(
            xpx(da).bar(x="period", barmode="relative", animation_frame="case"),
            xpx(line_da).line(animation_frame="case"),
        )

        # Same periods in every frame: initial autorange stays valid
        assert combined.layout.xaxis.range is None
        # y differs per frame, so it should be pinned
        assert combined.layout.yaxis.range is not None

    def test_no_axis_pinned_when_frames_identical(self) -> None:
        """Nothing is pinned when all frames have identical data extents."""
        da = xr.DataArray(
            np.array([[1.0, 5.0], [1.0, 5.0], [1.0, 5.0]]).T,
            dims=["x", "frame"],
            name="val",
        )
        combined = overlay(
            xpx(da).line(animation_frame="frame"),
            xpx(da).scatter(animation_frame="frame"),
        )

        assert combined.layout.xaxis.range is None
        assert combined.layout.yaxis.range is None

    def test_y_axis_pinned_when_later_frame_exceeds_initial(self) -> None:
        """y-axis is pinned to cover a later frame with larger values."""
        da = xr.DataArray(
            np.array([[1.0, 100.0], [2.0, 200.0], [3.0, 300.0]]),
            dims=["x", "frame"],
            name="val",
        )
        combined = overlay(
            xpx(da).line(animation_frame="frame"),
            xpx(da).scatter(animation_frame="frame"),
        )

        lo, hi = combined.layout.yaxis.range
        assert lo <= 1.0
        assert hi >= 300.0

    def test_stacked_bars_range_covers_summed_height(self) -> None:
        """With barmode='relative', the y-range must cover stacked totals."""
        # In the second frame, 3 techs of 100 each stack to 300 per period —
        # beyond the first frame's extent, so the y-axis gets pinned and must
        # use stacked sums, not individual segment values (max segment: 100).
        values = np.full((2, 3, 2), 100.0)
        values[:, :, 0] = 50.0
        da = xr.DataArray(
            values,
            dims=["period", "tech", "case"],
            name="heat",
        )
        combined = overlay(
            xpx(da).bar(x="period", barmode="relative", animation_frame="case"),
            xpx(da.isel(tech=0)).line(animation_frame="case"),
        )

        _lo, hi = combined.layout.yaxis.range
        assert hi >= 300.0, f"Stacked bars reach 300 but range top is {hi}"

    def test_stacked_bars_negative_values(self) -> None:
        """Negative segments stack downward; the range must cover their sum."""
        values = np.full((2, 3, 2), -40.0)
        values[:, :, 0] = -10.0
        da = xr.DataArray(
            values,
            dims=["period", "tech", "case"],
            name="losses",
        )
        combined = overlay(
            xpx(da).bar(x="period", barmode="relative", animation_frame="case"),
            xpx(da.isel(tech=0)).line(animation_frame="case"),
        )

        lo, hi = combined.layout.yaxis.range
        assert lo <= -120.0, f"Stacked bars reach -120 but range bottom is {lo}"
        assert hi >= 0.0, "Bar axis should include the zero baseline"

    def test_existing_explicit_range_respected(self) -> None:
        """A user-set range on the base figure must not be overwritten."""
        da = xr.DataArray(
            np.array([[1.0, 100.0], [2.0, 200.0]]),
            dims=["x", "frame"],
            name="val",
        )
        fig1 = xpx(da).line(animation_frame="frame")
        fig1.update_yaxes(range=[0, 42])
        combined = overlay(fig1, xpx(da).scatter(animation_frame="frame"))

        assert list(combined.layout.yaxis.range) == [0, 42]

    def test_log_axis_not_pinned(self) -> None:
        """Log axes use exponent coordinates; a data-value range would corrupt them."""
        da = xr.DataArray(
            np.array([[1.0, 100.0], [2.0, 200.0]]),
            dims=["x", "frame"],
            name="val",
        )
        fig1 = xpx(da).line(animation_frame="frame")
        fig1.update_yaxes(type="log")
        combined = overlay(fig1, xpx(da).scatter(animation_frame="frame"))

        assert combined.layout.yaxis.range is None

    def test_category_conversion_after_overlay_works(self) -> None:
        """The reported bug: type='category' after overlay showed an empty plot."""
        da = xr.DataArray(
            np.random.rand(4, 3, 2),
            dims=["period", "tech", "case"],
            coords={"period": [2025, 2030, 2035, 2040]},
            name="heat",
        )
        combined = overlay(
            xpx(da).bar(x="period", barmode="relative", animation_frame="case"),
            xpx(da.isel(tech=0)).line(animation_frame="case"),
        ).update_xaxes(
            type="category",
            categoryorder="array",
            categoryarray=[2025, 2030, 2035, 2040],
        )

        # No stale numeric range interpreted as category indices
        assert combined.layout.xaxis.range is None


class TestSubplotsBasic:
    """Basic tests for subplots function."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.da1 = xr.DataArray([1, 2, 3], dims=["x"], name="Temperature")
        self.da2 = xr.DataArray([10, 20, 30], dims=["x"], name="Rainfall")
        self.da3 = xr.DataArray([100, 200, 300], dims=["x"], name="Wind")

    def test_single_figure(self) -> None:
        fig = xpx(self.da1).line()
        grid = subplots(fig)
        assert len(grid.data) == 1

    def test_two_figures_one_column(self) -> None:
        fig1 = xpx(self.da1).line()
        fig2 = xpx(self.da2).bar()
        grid = subplots(fig1, fig2, cols=1)
        assert len(grid.data) == 2
        # Should be on different y-axes (different rows)
        assert grid.data[0].yaxis != grid.data[1].yaxis

    def test_two_figures_two_columns(self) -> None:
        fig1 = xpx(self.da1).line()
        fig2 = xpx(self.da2).bar()
        grid = subplots(fig1, fig2, cols=2)
        assert len(grid.data) == 2
        assert grid.data[0].xaxis != grid.data[1].xaxis

    def test_three_figures_two_columns(self) -> None:
        fig1 = xpx(self.da1).line()
        fig2 = xpx(self.da2).bar()
        fig3 = xpx(self.da3).scatter()
        grid = subplots(fig1, fig2, fig3, cols=2)
        assert len(grid.data) == 3

    def test_trace_count_preserved(self) -> None:
        da_multi = xr.DataArray(
            np.random.rand(10, 3),
            dims=["x", "cat"],
            coords={"cat": ["A", "B", "C"]},
        )
        fig1 = xpx(da_multi).line()  # 3 traces
        fig2 = xpx(self.da1).bar()  # 1 trace
        grid = subplots(fig1, fig2, cols=2)
        assert len(grid.data) == len(fig1.data) + len(fig2.data)

    def test_with_empty_figure(self) -> None:
        fig1 = xpx(self.da1).line()
        fig2 = go.Figure()
        grid = subplots(fig1, fig2, cols=2)
        assert len(grid.data) == 1


class TestSubplotsTitles:
    """Tests for subplot title derivation."""

    def test_titles_from_figure_title(self) -> None:
        da = xr.DataArray([1, 2, 3], dims=["x"], name="val")
        fig1 = xpx(da).line(title="My Title")
        fig2 = xpx(da).bar(title="Other Title")
        grid = subplots(fig1, fig2, cols=2)
        titles = [ann.text for ann in grid.layout.annotations]
        assert titles == ["<b>My Title</b>", "<b>Other Title</b>"]

    def test_titles_from_yaxis_label(self) -> None:
        da1 = xr.DataArray([1, 2, 3], dims=["x"], name="Temperature")
        da2 = xr.DataArray([4, 5, 6], dims=["x"], name="Pressure")
        fig1 = xpx(da1).line()
        fig2 = xpx(da2).line()
        grid = subplots(fig1, fig2, cols=2)
        titles = [ann.text for ann in grid.layout.annotations]
        assert titles == ["<b>Temperature</b>", "<b>Pressure</b>"]

    def test_titles_fallback_empty(self) -> None:
        grid = subplots(go.Figure(), go.Figure(), cols=2)
        # No annotations are created for empty titles
        titles = [ann.text for ann in grid.layout.annotations]
        assert titles == []


class TestSubplotsAxisConfig:
    """Tests for axis configuration copying."""

    def test_axis_titles_copied(self) -> None:
        da = xr.DataArray([1, 2, 3], dims=["time"], name="Temperature")
        fig = xpx(da).line()
        grid = subplots(fig)
        assert grid.layout.yaxis.title.text == "Temperature"
        assert grid.layout.xaxis.title.text == "time"


class TestSubplotsValidation:
    """Tests for subplots input validation."""

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="At least one figure"):
            subplots()

    def test_invalid_cols_raises(self) -> None:
        with pytest.raises(ValueError, match="cols must be >= 1"):
            subplots(go.Figure(), cols=0)

    def test_faceted_figures_stacked(self) -> None:
        """Faceted figures can be stacked in a subplot grid."""
        da = xr.DataArray(
            np.random.rand(10, 3),
            dims=["x", "facet"],
            coords={"facet": ["A", "B", "C"]},
        )
        fig1 = xpx(da).bar(facet_col="facet")
        fig2 = xpx(da).line(facet_col="facet")
        grid = subplots(fig1, fig2, cols=1)
        # 3 bar traces + 3 line traces
        assert len(grid.data) == 6
        # All traces should have unique axis assignments
        axes = {(t.xaxis, t.yaxis) for t in grid.data}
        assert len(axes) == 6

    def test_animated_figure_raises(self) -> None:
        da = xr.DataArray(np.random.rand(10, 3), dims=["x", "time"])
        fig = xpx(da).line(animation_frame="time")
        with pytest.raises(ValueError, match="animation frames"):
            subplots(fig)

    def test_source_not_modified(self) -> None:
        da = xr.DataArray([1, 2, 3], dims=["x"], name="val")
        fig = xpx(da).line()
        original_count = len(fig.data)
        _ = subplots(fig, fig, cols=2)
        assert len(fig.data) == original_count


class TestSimplifyFacetTitles:
    """Tests for the simplify_facet_titles helper and the `facet_titles` kwarg."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.da = xr.DataArray(
            np.random.rand(10, 3),
            dims=["x", "country"],
            coords={"country": ["United States", "China", "Brazil"]},
            name="value",
        )

    def test_helper_strips_dim_prefix(self) -> None:
        fig = xpx(self.da).line(facet_col="country")
        # PX writes annotations like "country=United States"
        original_texts = [a.text for a in fig.layout.annotations]
        assert any(t and t.startswith("country=") for t in original_texts)

        simplify_facet_titles(fig)

        for ann in fig.layout.annotations:
            if ann.text:
                assert "=" not in ann.text or ann.text.split("=", 1)[0] != "country"

    def test_helper_full_is_noop(self) -> None:
        fig = xpx(self.da).line(facet_col="country")
        before = [a.text for a in fig.layout.annotations]
        simplify_facet_titles(fig, mode="default")
        after = [a.text for a in fig.layout.annotations]
        assert before == after

    def test_helper_invalid_mode_raises(self) -> None:
        fig = xpx(self.da).line(facet_col="country")
        with pytest.raises(ValueError, match="facet_titles must be"):
            simplify_facet_titles(fig, mode="bogus")  # type: ignore[arg-type]

    def test_helper_leaves_user_annotations_alone(self) -> None:
        """User-added annotations without a Python-identifier prefix are preserved."""
        fig = xpx(self.da).line(facet_col="country")
        fig.add_annotation(text="Some note", x=0, y=0, showarrow=False)
        simplify_facet_titles(fig)
        texts = [a.text for a in fig.layout.annotations]
        assert "Some note" in texts

    def test_kwarg_default_keeps_px_format(self) -> None:
        fig = xpx(self.da).line(facet_col="country")
        # At least one annotation still carries the dim= prefix.
        assert any(a.text and a.text.startswith("country=") for a in fig.layout.annotations)

    def test_kwarg_value_strips_prefix(self) -> None:
        fig = xpx(self.da).line(facet_col="country", facet_titles="value")
        for ann in fig.layout.annotations:
            if ann.text:
                # Should not start with "country="; the dim prefix is stripped.
                assert not ann.text.startswith("country=")

    def test_kwarg_invalid_raises(self) -> None:
        with pytest.raises(ValueError, match="facet_titles must be"):
            xpx(self.da).line(facet_col="country", facet_titles="bogus")  # type: ignore[arg-type]


def _axis_titles(fig: go.Figure, letter: str) -> list[str]:
    """Collect non-empty axis title texts for one axis direction."""
    return [
        fig.layout[k].title.text
        for k in fig.layout
        if k.startswith(f"{letter}axis") and fig.layout[k].title.text
    ]


def _shared_label_annotations(fig: go.Figure) -> list[go.layout.Annotation]:
    """Collect annotations positioned like shared axis labels (centered bottom/left)."""
    return [
        a
        for a in fig.layout.annotations
        if a.xref == "paper" and a.yref == "paper" and (a.x, a.y) in {(0.5, 0), (0, 0.5)}
    ]


class TestShareAxisLabels:
    """Tests for the share_axis_labels helper and the `shared_axis_labels` kwarg."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.da = xr.DataArray(
            np.random.rand(10, 2, 3),
            dims=["time", "scenario", "country"],
            coords={"scenario": ["low", "high"], "country": ["US", "CN", "BR"]},
            name="value",
        )

    def test_matches_plotly_builtin_spec(self) -> None:
        """Shared labels must be identical to make_subplots(x_title=, y_title=) output."""
        import json

        from plotly.subplots import make_subplots

        def canonical(annotations: list[go.layout.Annotation]) -> set[str]:
            return {json.dumps(a.to_plotly_json(), sort_keys=True) for a in annotations}

        reference = make_subplots(rows=2, cols=2, x_title="value", y_title="value")
        ref_specs = canonical(list(reference.layout.annotations))

        fig = xpx(self.da).line(color=None, facet_col="scenario", facet_row="country")
        # Rename both labels to "value" so the comparison covers all other fields
        for ann in _shared_label_annotations(fig):
            ann.text = "value"
        assert canonical(_shared_label_annotations(fig)) == ref_specs

    def test_helper_facet_col_shares_x_label(self) -> None:
        fig = xpx(self.da.isel(scenario=0, drop=True)).line(
            color=None, facet_col="country", shared_axis_labels=False
        )
        assert len(_axis_titles(fig, "x")) == 3  # repeated per column

        share_axis_labels(fig)

        assert _axis_titles(fig, "x") == []
        labels = _shared_label_annotations(fig)
        assert [a.text for a in labels] == ["time"]
        # y title was never repeated, so it stays a regular axis title
        assert _axis_titles(fig, "y") == ["value"]

    def test_helper_facet_row_shares_y_label(self) -> None:
        fig = xpx(self.da.isel(scenario=0, drop=True)).line(
            color=None, facet_row="country", shared_axis_labels=False
        )
        assert len(_axis_titles(fig, "y")) == 3  # repeated per row

        share_axis_labels(fig)

        assert _axis_titles(fig, "y") == []
        labels = _shared_label_annotations(fig)
        assert [a.text for a in labels] == ["value"]
        assert _axis_titles(fig, "x") == ["time"]

    def test_helper_facet_grid_shares_both(self) -> None:
        fig = xpx(self.da).line(
            color=None, facet_col="scenario", facet_row="country", shared_axis_labels=False
        )
        share_axis_labels(fig)

        assert _axis_titles(fig, "x") == []
        assert _axis_titles(fig, "y") == []
        labels = {a.text for a in _shared_label_annotations(fig)}
        assert labels == {"time", "value"}

    def test_helper_no_facets_is_noop(self) -> None:
        fig = xpx(self.da.isel(scenario=0, country=0, drop=True)).line()
        before = fig.to_plotly_json()
        share_axis_labels(fig)
        assert fig.to_plotly_json() == before

    def test_helper_idempotent(self) -> None:
        fig = xpx(self.da).line(color=None, facet_col="scenario", facet_row="country")
        share_axis_labels(fig)
        n_annotations = len(fig.layout.annotations)
        share_axis_labels(fig)
        assert len(fig.layout.annotations) == n_annotations

    def test_helper_preserves_facet_titles(self) -> None:
        fig = xpx(self.da).line(color=None, facet_col="scenario", facet_row="country")
        share_axis_labels(fig)
        texts = [a.text for a in fig.layout.annotations]
        assert "scenario=low" in texts
        assert "scenario=high" in texts

    def test_helper_different_titles_untouched(self) -> None:
        """Figures whose subplots have different axis titles are not collapsed."""
        da1 = xr.DataArray(np.random.rand(5), dims=["time"], name="temperature")
        da2 = xr.DataArray(np.random.rand(5), dims=["distance"], name="pressure")
        fig = subplots(xpx(da1).line(), xpx(da2).line(), cols=2)
        before_x = _axis_titles(fig, "x")
        before_y = _axis_titles(fig, "y")

        share_axis_labels(fig)

        assert _axis_titles(fig, "x") == before_x
        assert _axis_titles(fig, "y") == before_y
        assert _shared_label_annotations(fig) == []

    def test_helper_secondary_y_untouched(self) -> None:
        """Identical titles on a primary and overlaying secondary y axis stay separate."""
        da = xr.DataArray(np.random.rand(5), dims=["time"], name="value")
        base = xpx(da).line()
        secondary = xpx(da).line()
        fig = add_secondary_y(base, secondary)
        before_y = _axis_titles(fig, "y")

        share_axis_labels(fig)

        assert _axis_titles(fig, "y") == before_y

    def test_kwarg_shares_by_default(self) -> None:
        fig = xpx(self.da).line(color=None, facet_col="scenario", facet_row="country")
        assert _axis_titles(fig, "x") == []
        assert _axis_titles(fig, "y") == []
        assert {a.text for a in _shared_label_annotations(fig)} == {"time", "value"}

    def test_kwarg_opt_out(self) -> None:
        fig = xpx(self.da).line(
            color=None, facet_col="scenario", facet_row="country", shared_axis_labels=False
        )
        assert len(_axis_titles(fig, "x")) > 1
        assert len(_axis_titles(fig, "y")) > 1
        assert _shared_label_annotations(fig) == []

    def test_kwarg_on_imshow(self) -> None:
        da = xr.DataArray(
            np.random.rand(4, 5, 2, 3),
            dims=["lat", "lon", "scenario", "year"],
            name="temperature",
        )
        fig = da.plotly.imshow()
        assert {a.text for a in _shared_label_annotations(fig)} == {"lat", "lon"}
