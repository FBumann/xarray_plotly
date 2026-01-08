# xarray_plotly

**Interactive Plotly Express plotting accessor for xarray**

xarray_plotly provides a `plotly` accessor for xarray DataArray objects that enables interactive plotting using Plotly Express. It automatically assigns dimensions to plot slots based on their order, making it easy to create rich, interactive visualizations with minimal code.

## Features

- **Interactive plots**: Zoom, pan, hover for values, toggle traces - all built-in
- **Automatic dimension assignment**: Dimensions fill plot slots (x, color, facet_col, etc.) by position
- **Easy customization**: Returns Plotly `Figure` objects for further modification
- **Multiple plot types**: Line, bar, area, scatter, box, and heatmap plots
- **Faceting and animation**: Built-in support for subplot grids and animated time series

## Quick Example

```python
import xarray as xr
import numpy as np
import xarray_plotly  # registers the accessor

# Create sample data
da = xr.DataArray(
    np.random.randn(100, 3, 2),
    dims=["time", "city", "scenario"],
    coords={
        "time": np.arange(100),
        "city": ["NYC", "LA", "Chicago"],
        "scenario": ["baseline", "warming"],
    },
    name="temperature",
)

# Create an interactive line plot
# Dimensions auto-assign: time->x, city->color, scenario->facet_col
fig = da.plotly.line()
fig.show()

# Easy customization
fig.update_layout(
    title="Temperature Projections",
    template="plotly_dark",
)
```

## Installation

```bash
pip install xarray_plotly
```

Or with uv:

```bash
uv add xarray_plotly
```

## Why xarray_plotly?

The current `.plot` accessor in xarray is built on matplotlib, which has limitations for modern data exploration:

1. **Static outputs**: Matplotlib plots are non-interactive
2. **Post-creation modification is cumbersome**: Requires understanding complex object hierarchies
3. **Multi-dimensional data**: No built-in support for faceting or animation

xarray_plotly solves these with Plotly Express, providing:

- Interactive plots with zero additional code
- Simple, predictable dimension-to-slot assignment
- Easy post-creation customization via Plotly's `Figure` API
- Modern visualization patterns (faceting, animation) built-in
