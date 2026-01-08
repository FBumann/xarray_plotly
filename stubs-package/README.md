# xarray_plotly_stubs

Type stubs for [xarray_plotly](https://github.com/felix/xarray_plotly).

## Installation

```bash
pip install xarray_plotly_stubs
```

This package provides type hints for the `da.plotly` accessor added by xarray_plotly.

## Usage

After installing, type checkers will recognize the `plotly` accessor on xarray DataArrays:

```python
import xarray as xr
import xarray_plotly  # registers the accessor

da = xr.DataArray(...)
fig = da.plotly.line()  # Now properly typed!
```

## Note

Install this package alongside `xarray_plotly` for full type checking support.
