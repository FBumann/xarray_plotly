# xarray_plotly

**Interactive Plotly Express plotting for xarray**

[![PyPI version](https://badge.fury.io/py/xarray_plotly.svg)](https://badge.fury.io/py/xarray_plotly)
[![Python](https://img.shields.io/pypi/pyversions/xarray_plotly.svg)](https://pypi.org/project/xarray_plotly/)
[![Docs](https://img.shields.io/badge/docs-fbumann.github.io-blue)](https://fbumann.github.io/xarray_plotly/)

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

## Documentation

Full documentation: [https://fbumann.github.io/xarray_plotly](https://fbumann.github.io/xarray_plotly)

## License

MIT
