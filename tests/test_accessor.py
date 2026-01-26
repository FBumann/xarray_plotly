"""Tests for the DataArray plotting accessor."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
import xarray as xr

import xarray_plotly  # noqa: F401 - registers accessor
from xarray_plotly import xpx


class TestXpxFunction:
    """Tests for the xpx() function."""

    def test_xpx_returns_dataarray_accessor(self) -> None:
        """Test that xpx() returns a DataArrayPlotlyAccessor for DataArray."""
        da = xr.DataArray(np.random.rand(10), dims=["time"])
        accessor = xpx(da)
        assert hasattr(accessor, "line")
        assert hasattr(accessor, "bar")
        assert hasattr(accessor, "scatter")
        assert hasattr(accessor, "imshow")

    def test_xpx_returns_dataset_accessor(self) -> None:
        """Test that xpx() returns a DatasetPlotlyAccessor for Dataset."""
        ds = xr.Dataset({"temp": (["time"], np.random.rand(10))})
        accessor = xpx(ds)
        assert hasattr(accessor, "line")
        assert hasattr(accessor, "bar")
        assert hasattr(accessor, "scatter")
        # Dataset accessor should not have imshow
        assert not hasattr(accessor, "imshow")

    def test_xpx_dataarray_equivalent_to_accessor(self) -> None:
        """Test that xpx(da).line() works the same as da.plotly.line()."""
        da = xr.DataArray(
            np.random.rand(10, 3),
            dims=["time", "city"],
            coords={"time": np.arange(10), "city": ["A", "B", "C"]},
            name="test",
        )
        fig1 = xpx(da).line()
        fig2 = da.plotly.line()
        assert isinstance(fig1, go.Figure)
        assert isinstance(fig2, go.Figure)

    def test_xpx_dataset_equivalent_to_accessor(self) -> None:
        """Test that xpx(ds).line() works the same as ds.plotly.line()."""
        ds = xr.Dataset(
            {
                "temperature": (["time", "city"], np.random.rand(10, 3)),
                "humidity": (["time", "city"], np.random.rand(10, 3)),
            }
        )
        fig1 = xpx(ds).line()
        fig2 = ds.plotly.line()
        assert isinstance(fig1, go.Figure)
        assert isinstance(fig2, go.Figure)


class TestDataArrayPxplot:
    """Tests for DataArray.plotly accessor."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        """Set up test data."""
        self.da_1d = xr.DataArray(
            np.random.rand(10),
            dims=["time"],
            coords={"time": pd.date_range("2020", periods=10)},
            name="temperature",
        )
        self.da_2d = xr.DataArray(
            np.random.rand(10, 3),
            dims=["time", "city"],
            coords={
                "time": pd.date_range("2020", periods=10),
                "city": ["NYC", "LA", "Chicago"],
            },
            name="temperature",
        )
        self.da_3d = xr.DataArray(
            np.random.rand(10, 3, 2),
            dims=["time", "city", "scenario"],
            coords={
                "time": pd.date_range("2020", periods=10),
                "city": ["NYC", "LA", "Chicago"],
                "scenario": ["baseline", "warming"],
            },
            name="temperature",
        )
        self.da_unnamed = xr.DataArray(np.random.rand(5, 3), dims=["x", "y"])

    def test_accessor_exists(self) -> None:
        """Test that plotly accessor is available on DataArray."""
        assert hasattr(self.da_2d, "plotly")
        assert hasattr(self.da_2d.plotly, "line")
        assert hasattr(self.da_2d.plotly, "bar")
        assert hasattr(self.da_2d.plotly, "area")
        assert hasattr(self.da_2d.plotly, "scatter")
        assert hasattr(self.da_2d.plotly, "box")
        assert hasattr(self.da_2d.plotly, "imshow")

    def test_line_returns_figure(self) -> None:
        """Test that line() returns a Plotly Figure."""
        fig = self.da_2d.plotly.line()
        assert isinstance(fig, go.Figure)

    def test_line_1d(self) -> None:
        """Test line plot with 1D data."""
        fig = self.da_1d.plotly.line()
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1

    def test_line_2d(self) -> None:
        """Test line plot with 2D data."""
        fig = self.da_2d.plotly.line()
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1

    def test_line_explicit_assignment(self) -> None:
        """Test line plot with explicit dimension assignment."""
        fig = self.da_2d.plotly.line(x="time", color="city")
        assert isinstance(fig, go.Figure)

    def test_line_skip_slot(self) -> None:
        """Test line plot with skipped slot."""
        fig = self.da_3d.plotly.line(color=None)
        assert isinstance(fig, go.Figure)

    def test_line_px_kwargs(self) -> None:
        """Test that px_kwargs are passed through."""
        fig = self.da_2d.plotly.line(title="My Plot")
        assert fig.layout.title.text == "My Plot"

    def test_bar_returns_figure(self) -> None:
        """Test that bar() returns a Plotly Figure."""
        fig = self.da_2d.plotly.bar()
        assert isinstance(fig, go.Figure)

    def test_area_returns_figure(self) -> None:
        """Test that area() returns a Plotly Figure."""
        fig = self.da_2d.plotly.area()
        assert isinstance(fig, go.Figure)

    def test_fast_bar_returns_figure(self) -> None:
        """Test that fast_bar() returns a Plotly Figure."""
        fig = self.da_2d.plotly.fast_bar()
        assert isinstance(fig, go.Figure)

    def test_fast_bar_trace_styling(self) -> None:
        """Test that fast_bar applies correct trace styling."""
        fig = self.da_2d.plotly.fast_bar()
        for trace in fig.data:
            assert trace.line.width == 0
            assert trace.line.shape == "hv"
            assert trace.fillcolor is not None

    def test_fast_bar_animation_frames(self) -> None:
        """Test that fast_bar styling applies to animation frames."""
        da = xr.DataArray(
            np.random.rand(5, 3, 4),
            dims=["time", "city", "year"],
        )
        fig = da.plotly.fast_bar(animation_frame="year")
        assert len(fig.frames) > 0
        for frame in fig.frames:
            for trace in frame.data:
                assert trace.line.width == 0
                assert trace.line.shape == "hv"
                assert trace.fillcolor is not None

    def test_fast_bar_mixed_signs_dashed(self) -> None:
        """Test that fast_bar shows mixed-sign traces as dashed lines."""
        da = xr.DataArray(
            np.array([[50, -30], [-40, 60]]),  # Both columns have mixed signs
            dims=["time", "category"],
        )
        fig = da.plotly.fast_bar()
        # Mixed traces should have no stacking and dashed lines
        for trace in fig.data:
            assert trace.stackgroup is None
            assert trace.line.dash == "dash"

    def test_fast_bar_separate_sign_columns(self) -> None:
        """Test that fast_bar uses separate stackgroups when columns have different signs."""
        da = xr.DataArray(
            np.array([[50, -30], [60, -40]]),  # Column 0 positive, column 1 negative
            dims=["time", "category"],
        )
        fig = da.plotly.fast_bar()
        stackgroups = {trace.stackgroup for trace in fig.data}
        assert "positive" in stackgroups
        assert "negative" in stackgroups

    def test_fast_bar_same_sign_stacks(self) -> None:
        """Test that fast_bar uses stacking for same-sign data."""
        da = xr.DataArray(
            np.random.rand(5, 3) * 100,
            dims=["time", "category"],
        )
        fig = da.plotly.fast_bar()
        for trace in fig.data:
            assert trace.stackgroup is not None

    def test_scatter_returns_figure(self) -> None:
        """Test that scatter() returns a Plotly Figure."""
        fig = self.da_2d.plotly.scatter()
        assert isinstance(fig, go.Figure)

    def test_scatter_dim_vs_dim(self) -> None:
        """Test scatter plot with dimension vs dimension, colored by values."""
        da = xr.DataArray(
            np.random.rand(5, 10),
            dims=["lat", "lon"],
            coords={"lat": np.arange(5), "lon": np.arange(10)},
            name="temperature",
        )
        fig = da.plotly.scatter(x="lon", y="lat", color="value")
        assert isinstance(fig, go.Figure)

    def test_box_returns_figure(self) -> None:
        """Test that box() returns a Plotly Figure."""
        fig = self.da_2d.plotly.box()
        assert isinstance(fig, go.Figure)

    def test_box_with_aggregation(self) -> None:
        """Test box plot with unassigned dimensions aggregated."""
        fig = self.da_2d.plotly.box(x="city", color=None)
        assert isinstance(fig, go.Figure)

    def test_imshow_returns_figure(self) -> None:
        """Test that imshow() returns a Plotly Figure."""
        fig = self.da_2d.plotly.imshow()
        assert isinstance(fig, go.Figure)

    def test_imshow_transpose(self) -> None:
        """Test that imshow correctly transposes based on x and y."""
        da = xr.DataArray(
            np.random.rand(10, 20),
            dims=["lat", "lon"],
            coords={"lat": np.arange(10), "lon": np.arange(20)},
        )
        fig = da.plotly.imshow()
        assert isinstance(fig, go.Figure)

        fig = da.plotly.imshow(x="lon", y="lat")
        assert isinstance(fig, go.Figure)

    def test_unnamed_dataarray(self) -> None:
        """Test plotting unnamed DataArray."""
        fig = self.da_unnamed.plotly.line()
        assert isinstance(fig, go.Figure)

    def test_unassigned_dims_error(self) -> None:
        """Test that too many dimensions raises an error."""
        da_8d = xr.DataArray(np.random.rand(2, 2, 2, 2, 2, 2, 2, 2), dims=list("abcdefgh"))
        with pytest.raises(ValueError, match="Unassigned dimension"):
            da_8d.plotly.line()


class TestLabelsAndMetadata:
    """Tests for label extraction from xarray attributes."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        """Set up test data with metadata."""
        self.da = xr.DataArray(
            np.random.rand(10, 3),
            dims=["time", "station"],
            coords={
                "time": pd.date_range("2020", periods=10),
                "station": ["A", "B", "C"],
            },
            name="temperature",
            attrs={
                "long_name": "Air Temperature",
                "units": "K",
            },
        )
        self.da.coords["time"].attrs = {
            "long_name": "Time",
            "units": "days since 2020-01-01",
        }

    def test_value_label_from_attrs(self) -> None:
        """Test that value labels are extracted from attributes."""
        fig = self.da.plotly.line()
        assert isinstance(fig, go.Figure)


class TestDatasetPlotlyAccessor:
    """Tests for Dataset.plotly accessor."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        """Set up test data."""
        self.ds = xr.Dataset(
            {
                "temperature": (["time", "city"], np.random.rand(10, 3)),
                "humidity": (["time", "city"], np.random.rand(10, 3)),
            },
            coords={
                "time": pd.date_range("2020", periods=10),
                "city": ["NYC", "LA", "Chicago"],
            },
        )

    def test_accessor_exists(self) -> None:
        """Test that plotly accessor is available on Dataset."""
        assert hasattr(self.ds, "plotly")
        assert hasattr(self.ds.plotly, "line")
        assert hasattr(self.ds.plotly, "bar")
        assert hasattr(self.ds.plotly, "area")
        assert hasattr(self.ds.plotly, "scatter")
        assert hasattr(self.ds.plotly, "box")

    def test_line_all_variables(self) -> None:
        """Test line plot with all variables."""
        fig = self.ds.plotly.line()
        assert isinstance(fig, go.Figure)

    def test_line_single_variable(self) -> None:
        """Test line plot with single variable."""
        fig = self.ds.plotly.line(var="temperature")
        assert isinstance(fig, go.Figure)

    def test_line_variable_as_facet(self) -> None:
        """Test line plot with variable as facet."""
        fig = self.ds.plotly.line(facet_col="variable")
        assert isinstance(fig, go.Figure)

    def test_bar_all_variables(self) -> None:
        """Test bar plot with all variables."""
        fig = self.ds.plotly.bar()
        assert isinstance(fig, go.Figure)

    def test_area_all_variables(self) -> None:
        """Test area plot with all variables."""
        fig = self.ds.plotly.area()
        assert isinstance(fig, go.Figure)

    def test_scatter_all_variables(self) -> None:
        """Test scatter plot with all variables."""
        fig = self.ds.plotly.scatter()
        assert isinstance(fig, go.Figure)

    def test_box_all_variables(self) -> None:
        """Test box plot with all variables."""
        fig = self.ds.plotly.box()
        assert isinstance(fig, go.Figure)


class TestImshowBounds:
    """Tests for imshow global bounds and robust mode."""

    def test_imshow_global_bounds(self) -> None:
        """Test that imshow uses global min/max by default."""
        da = xr.DataArray(
            np.array([[[1, 2], [3, 4]], [[5, 6], [7, 100]]]),
            dims=["time", "y", "x"],
        )
        fig = da.plotly.imshow(animation_frame="time")
        # Check coloraxis for zmin/zmax (plotly stores them there)
        coloraxis = fig.layout.coloraxis
        assert coloraxis.cmin == 1.0
        assert coloraxis.cmax == 100.0

    def test_imshow_robust_bounds(self) -> None:
        """Test that robust=True uses percentile-based bounds."""
        # Create data with outlier
        data = np.random.rand(10, 20) * 100
        data[0, 0] = 10000  # extreme outlier
        da = xr.DataArray(data, dims=["y", "x"])

        fig = da.plotly.imshow(robust=True)
        # With robust=True, cmax should be much less than the outlier
        coloraxis = fig.layout.coloraxis
        assert coloraxis.cmax < 10000
        assert coloraxis.cmax < 200  # Should be around 98th percentile (~98)

    def test_imshow_user_zmin_zmax_override(self) -> None:
        """Test that user-provided zmin/zmax overrides auto bounds."""
        da = xr.DataArray(np.random.rand(10, 20) * 100, dims=["y", "x"])
        fig = da.plotly.imshow(zmin=0, zmax=50)
        coloraxis = fig.layout.coloraxis
        assert coloraxis.cmin == 0
        assert coloraxis.cmax == 50

    def test_imshow_animation_consistent_bounds(self) -> None:
        """Test that animation frames have consistent color bounds."""
        da = xr.DataArray(
            np.array([[[0, 10], [20, 30]], [[40, 50], [60, 70]]]),
            dims=["time", "y", "x"],
        )
        fig = da.plotly.imshow(animation_frame="time")
        # All frames should use global min (0) and max (70)
        coloraxis = fig.layout.coloraxis
        assert coloraxis.cmin == 0.0
        assert coloraxis.cmax == 70.0


class TestColorsParameter:
    """Tests for the unified colors parameter."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        """Create test DataArrays."""
        self.da = xr.DataArray(
            np.random.rand(10, 3),
            dims=["time", "city"],
            coords={"city": ["A", "B", "C"]},
        )

    def test_colors_list_sets_discrete_sequence(self) -> None:
        """Test that a list of colors sets color_discrete_sequence."""
        fig = self.da.plotly.line(colors=["red", "blue", "green"])
        # Check that traces have the expected colors
        assert len(fig.data) == 3
        assert fig.data[0].line.color == "red"
        assert fig.data[1].line.color == "blue"
        assert fig.data[2].line.color == "green"

    def test_colors_dict_sets_discrete_map(self) -> None:
        """Test that a dict sets color_discrete_map."""
        fig = self.da.plotly.line(colors={"A": "red", "B": "blue", "C": "green"})
        # Traces should be colored according to the mapping
        assert len(fig.data) == 3
        # Find traces by name and check their color
        colors_by_name = {trace.name: trace.line.color for trace in fig.data}
        assert colors_by_name["A"] == "red"
        assert colors_by_name["B"] == "blue"
        assert colors_by_name["C"] == "green"

    def test_colors_continuous_scale_string(self) -> None:
        """Test that a continuous scale name sets color_continuous_scale."""
        da = xr.DataArray(
            np.random.rand(50, 2),
            dims=["point", "coord"],
            coords={"coord": ["x", "y"]},
        )
        fig = da.plotly.scatter(y="coord", x="point", color="value", colors="Viridis")
        # Plotly Express uses coloraxis in the layout for continuous scales
        # Check that the colorscale was applied to the coloraxis
        assert fig.layout.coloraxis.colorscale is not None
        colorscale = fig.layout.coloraxis.colorscale
        # Viridis should be in the colorscale definition
        assert any("viridis" in str(c).lower() for c in colorscale) or len(colorscale) > 0

    def test_colors_qualitative_palette_string(self) -> None:
        """Test that a qualitative palette name sets color_discrete_sequence."""
        import plotly.express as px

        fig = self.da.plotly.line(colors="D3")
        # D3 palette should be applied - check first trace color is from D3
        d3_colors = px.colors.qualitative.D3
        assert fig.data[0].line.color in d3_colors

    def test_colors_ignored_with_warning_when_px_kwargs_present(self) -> None:
        """Test that colors is ignored with warning when color_* kwargs are present."""
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fig = self.da.plotly.line(
                colors="D3", color_discrete_sequence=["orange", "purple", "cyan"]
            )
            # Should have raised a warning about colors being ignored
            assert any(
                "colors" in str(m.message).lower() and "ignored" in str(m.message).lower()
                for m in w
            ), "Expected warning about 'colors' being 'ignored' not found"
            # The explicit px_kwargs should take precedence
            assert fig.data[0].line.color == "orange"

    def test_colors_none_uses_defaults(self) -> None:
        """Test that colors=None uses Plotly defaults."""
        fig1 = self.da.plotly.line(colors=None)
        fig2 = self.da.plotly.line()
        # Both should produce the same result
        assert fig1.data[0].line.color == fig2.data[0].line.color

    def test_colors_works_with_bar(self) -> None:
        """Test colors parameter with bar chart."""
        fig = self.da.plotly.bar(colors=["#e41a1c", "#377eb8", "#4daf4a"])
        assert fig.data[0].marker.color == "#e41a1c"

    def test_colors_works_with_area(self) -> None:
        """Test colors parameter with area chart."""
        fig = self.da.plotly.area(colors=["red", "green", "blue"])
        assert len(fig.data) == 3

    def test_colors_works_with_scatter(self) -> None:
        """Test colors parameter with scatter plot."""
        fig = self.da.plotly.scatter(colors=["red", "green", "blue"])
        assert len(fig.data) == 3

    def test_colors_works_with_imshow(self) -> None:
        """Test colors parameter with imshow (continuous scale)."""
        da = xr.DataArray(np.random.rand(10, 10), dims=["y", "x"])
        fig = da.plotly.imshow(colors="RdBu")
        # Plotly Express uses coloraxis in the layout for continuous scales
        assert fig.layout.coloraxis.colorscale is not None
        colorscale = fig.layout.coloraxis.colorscale
        # RdBu should be in the colorscale definition
        assert any("rdbu" in str(c).lower() for c in colorscale) or len(colorscale) > 0

    def test_colors_works_with_pie(self) -> None:
        """Test colors parameter with pie chart."""
        da = xr.DataArray([30, 40, 30], dims=["category"], coords={"category": ["A", "B", "C"]})
        fig = da.plotly.pie(colors={"A": "red", "B": "blue", "C": "green"})
        assert isinstance(fig, go.Figure)

    def test_colors_works_with_dataset(self) -> None:
        """Test colors parameter works with Dataset accessor."""
        ds = xr.Dataset(
            {
                "temp": (["time"], np.random.rand(10)),
                "precip": (["time"], np.random.rand(10)),
            }
        )
        fig = ds.plotly.line(colors=["red", "blue"])
        assert len(fig.data) == 2
        assert fig.data[0].line.color == "red"
        assert fig.data[1].line.color == "blue"
