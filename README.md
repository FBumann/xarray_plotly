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
from xarray_plotly import xpx

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
fig = xpx(da).line()
fig.show()

# Easy customization
fig.update_layout(title="Temperature Projections", template="plotly_dark")
```

## Usage Styles

xarray_plotly supports two equivalent usage styles:

```python
# Function style (recommended) - full IDE code completion
from xarray_plotly import xpx
fig = xpx(da).line()

# Accessor style - works but no IDE completion
import xarray_plotly
fig = da.plotly.line()
```

The `xpx()` function is recommended as it provides full IDE code completion and type hints.

**Why no IDE completion for the accessor?** xarray accessors are registered dynamically at runtime using `register_dataarray_accessor()`. Python's static type checkers and IDEs cannot see these dynamically added attributes, so `da.plotly` appears as an unknown attribute. This limitation could only be solved by xarray itself (e.g., through a type checker plugin), if at all. The `xpx()` function provides a workaround with an explicit return type that IDEs can follow.

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
xpx(da).line()

# Override with explicit assignments
xpx(da).line(x="time", color="scenario", facet_col="city")

# Skip a slot with None
xpx(da).line(color=None)  # time→x, city→facet_col
```

## Available Methods

| Method | Description | Slot Order |
|--------|-------------|------------|
| `line()` | Line plot | x → color → line_dash → symbol → facet_col → facet_row → animation_frame |
| `bar()` | Bar chart | x → color → pattern_shape → facet_col → facet_row → animation_frame |
| `area()` | Stacked area | x → color → pattern_shape → facet_col → facet_row → animation_frame |
| `scatter()` | Scatter plot | x → color → symbol → facet_col → facet_row → animation_frame |
| `box()` | Box plot | x → color → facet_col → facet_row → animation_frame |
| `imshow()` | Heatmap | y → x → facet_col → animation_frame |

## Configuration

Customize label extraction and slot assignment behavior:

```python
import xarray_plotly as xp

# View current options
xp.config.get_options()

# Set globally
with xp.config.set_options(label_include_units=False):
    fig = xpx(da).line()  # Labels won't include units
```

**Available options:**

| Option | Default | Description |
|--------|---------|-------------|
| `label_use_long_name` | `True` | Use `long_name` attr for labels |
| `label_use_standard_name` | `True` | Fall back to `standard_name` |
| `label_include_units` | `True` | Append units to labels |
| `label_unit_format` | `"[{units}]"` | Format string for units |
| `slot_orders` | (defaults) | Slot orders per plot type |

## Documentation

Full documentation with examples: [https://felix.github.io/xarray_plotly](https://felix.github.io/xarray_plotly)

## License

MIT
