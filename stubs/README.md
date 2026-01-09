# xarray-plotly-stubs

Type stubs that add the `plotly` accessor to `xarray.DataArray` for IDE code completion.

This package is automatically installed with `xarray-plotly` and provides type information so that IDEs like PyCharm and VS Code can offer code completion for:

```python
import xarray as xr
import xarray_plotly

da = xr.DataArray(...)
da.plotly.line()  # IDE completion works here
```

## Installation

This package is installed automatically as a dependency of `xarray-plotly`. You don't need to install it separately.
