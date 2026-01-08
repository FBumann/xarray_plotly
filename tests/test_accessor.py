"""Tests for the DataArray plotting accessor."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
import xarray as xr

import xarray_plotly  # noqa: F401 - registers accessor


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
