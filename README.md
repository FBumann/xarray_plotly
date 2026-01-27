# xarray_plotly

**Interactive Plotly Express plotting for xarray**

[![PyPI version](https://badge.fury.io/py/xarray-plotly.svg)](https://badge.fury.io/py/xarray-plotly)
[![Python](https://img.shields.io/pypi/pyversions/xarray-plotly.svg)](https://pypi.org/project/xarray-plotly/)
[![CI](https://github.com/FBumann/xarray_plotly/actions/workflows/ci.yml/badge.svg)](https://github.com/FBumann/xarray_plotly/actions)
[![Docs](https://img.shields.io/badge/docs-fbumann.github.io-blue)](https://fbumann.github.io/xarray_plotly/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Installation

```bash
pip install xarray_plotly
```

## Quick Start

```python
import xarray as xr
import numpy as np
import xarray_plotly  # registers the accessor

da = xr.DataArray(
    np.random.randn(100, 3).cumsum(axis=0),
    dims=["time", "city"],
    coords={"time": np.arange(100), "city": ["NYC", "LA", "Chicago"]},
)

# Accessor style
fig = da.plotly.line()
fig.show()

# Or with xpx() for IDE code completion
from xarray_plotly import xpx
fig = xpx(da).line()
```

**Why `xpx()`?** The accessor (`da.plotly`) works but IDEs can't provide code completion for it. This is because xarray accessors are registered dynamically at runtime, making them invisible to static type checkers. The `xpx()` function provides the same functionality with full IDE support. This limitation could only be solved by xarray itself, if at all — it may be a fundamental Python limitation.

## Documentation

Full documentation: [https://fbumann.github.io/xarray_plotly](https://fbumann.github.io/xarray_plotly)

## Roadmap

Planned additions (contributions welcome):

**New plot types**
- `histogram()` — distribution of DataArray values
- `violin()` — richer distribution visualization than box plots
- `density_heatmap()` — 2D histograms (x and y as dimensions)
- `density_contour()` — 2D density contours (x and y as dimensions)

**Enhancements**
- WebGL rendering option for `line()` and `scatter()` (large datasets)

**Figure utilities** (facet/animation-aware)
- `add_trace()` — add a trace to base figure and all animation frames
- `fill_between()` — fill area between two traces (uncertainty bands)
- `sync_axes()` — consistent axis ranges across facets and animation frames
- `add_secondary_x()` — secondary x-axis (like `add_secondary_y()`)
- `stack()` — vertically stack separate figures into subplots

## License

MIT
