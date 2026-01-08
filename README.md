# xarray_plotly

**Interactive Plotly Express plotting accessor for xarray**

[![PyPI version](https://badge.fury.io/py/xarray_plotly.svg)](https://badge.fury.io/py/xarray_plotly)
[![Python](https://img.shields.io/pypi/pyversions/xarray_plotly.svg)](https://pypi.org/project/xarray_plotly/)

xarray_plotly provides a `plotly` accessor for xarray DataArray objects that enables interactive plotting using Plotly Express with automatic dimension-to-slot assignment.

## Installation

```bash
pip install xarray_plotly
```

Or with uv:

```bash
uv add xarray_plotly
```

## Quick Start

```python
import xarray as xr
import numpy as np
import xarray_plotly  # registers the accessor

# Create sample data
da = xr.DataArray(
    np.random.randn(100, 3, 2).cumsum(axis=0),
    dims=["time", "city", "scenario"],
    coords={
        "time": np.arange(100),
        "city": ["NYC", "LA", "Chicago"],
        "scenario": ["baseline", "warming"],
    },
    name="temperature",
)

# Create an interactive line plot
# Dimensions auto-assign: time→x, city→color, scenario→facet_col
fig = da.plotly.line()
fig.show()

# Easy customization
fig.update_layout(title="Temperature Projections", template="plotly_dark")
```

## Features

- **Interactive plots**: Zoom, pan, hover for values, toggle traces
- **Automatic dimension assignment**: Dimensions fill plot slots by position
- **Easy customization**: Returns Plotly `Figure` objects
- **Multiple plot types**: `line()`, `bar()`, `area()`, `scatter()`, `box()`, `imshow()`
- **Faceting and animation**: Built-in support for subplot grids and animations

## Dimension Assignment

Dimensions are automatically assigned to plot "slots" based on their order:

```python
# dims: (time, city, scenario)
# auto-assigns: time→x, city→color, scenario→facet_col
da.plotly.line()

# Override with explicit assignments
da.plotly.line(x="time", color="scenario", facet_col="city")

# Skip a slot with None
da.plotly.line(color=None)  # time→x, city→facet_col
```

## Available Methods

| Method | Description | Slot Order |
|--------|-------------|------------|
| `line()` | Line plot | x → color → line_dash → symbol → facet_col → facet_row → animation_frame |
| `bar()` | Bar chart | x → color → pattern_shape → facet_col → facet_row → animation_frame |
| `area()` | Stacked area | x → color → pattern_shape → facet_col → facet_row → animation_frame |
| `scatter()` | Scatter plot | x → color → size → symbol → facet_col → facet_row → animation_frame |
| `box()` | Box plot | x → color → facet_col → facet_row → animation_frame |
| `imshow()` | Heatmap | y → x → facet_col → animation_frame |

## Documentation

Full documentation with examples: [https://felix.github.io/xarray_plotly](https://felix.github.io/xarray_plotly)

## License

MIT
