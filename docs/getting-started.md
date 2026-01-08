# Getting Started

## Installation

Install xarray-plotly using pip:

```bash
pip install xarray-plotly
```

Or with uv:

```bash
uv add xarray-plotly
```

## Basic Usage

Import the package to register the `pxplot` accessor on xarray DataArray objects:

```python
import xarray as xr
import numpy as np
import xarray_plotly  # This registers the accessor

# Create a DataArray
da = xr.DataArray(
    np.random.rand(10, 3),
    dims=["time", "city"],
    coords={
        "time": np.arange(10),
        "city": ["NYC", "LA", "Chicago"],
    },
    name="temperature",
)

# Create a line plot - dimensions auto-assign to slots
fig = da.pxplot.line()
fig.show()
```

## Available Plot Types

The `pxplot` accessor provides these plotting methods:

| Method | Description |
|--------|-------------|
| `line()` | Interactive line plot |
| `bar()` | Interactive bar chart |
| `area()` | Stacked area chart |
| `scatter()` | Scatter plot |
| `box()` | Box plot |
| `imshow()` | Heatmap image |

## Dimension Assignment

Dimensions are automatically assigned to plot "slots" based on their order:

```python
# DataArray with dims: (time, city, scenario)
da = xr.DataArray(..., dims=["time", "city", "scenario"])

# Auto-assigns: time->x, city->color, scenario->facet_col
fig = da.pxplot.line()
```

You can override the automatic assignment:

```python
# Explicit assignment
fig = da.pxplot.line(x="time", color="scenario", facet_col="city")

# Skip a slot with None
fig = da.pxplot.line(color=None)  # time->x, city->facet_col
```

## Customizing Plots

All methods return a Plotly `Figure` object that you can customize:

```python
fig = da.pxplot.line()

# Modify layout
fig.update_layout(
    title="My Custom Title",
    xaxis_title="Date",
    template="plotly_dark",
)

# Show the plot
fig.show()
```

You can also pass Plotly Express arguments directly:

```python
fig = da.pxplot.line(
    title="Temperature Over Time",
    color_discrete_sequence=["red", "blue", "green"],
)
```

## Next Steps

- Learn about [Dimension Assignment](guide/dimension-assignment.md) in detail
- Explore all [Plot Types](guide/plot-types.md)
- See [Customization](guide/customization.md) options
