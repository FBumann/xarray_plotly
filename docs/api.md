# API Reference

## xpx Function

The recommended way to use xarray_plotly with full IDE code completion:

```python
from xarray_plotly import xpx

fig = xpx(da).line()
```

::: xarray_plotly.xpx
    options:
      show_root_heading: true

## Accessor

The accessor style (`da.plotly.line()`) works but doesn't provide IDE code completion.

::: xarray_plotly.accessor.DataArrayPlotlyAccessor
    options:
      show_root_heading: true
      members:
        - line
        - bar
        - area
        - scatter
        - box
        - imshow

## Plotting Functions

::: xarray_plotly.plotting
    options:
      show_root_heading: true
      members:
        - line
        - bar
        - area
        - scatter
        - box
        - imshow

## Configuration

Customize label extraction and slot assignment behavior:

```python
from xarray_plotly import config, xpx

# View current options
config.get_options()

# Set options (works as context manager)
with config.set_options(label_include_units=False):
    fig = xpx(da).line()
```

::: xarray_plotly.config
    options:
      show_root_heading: true
      members:
        - notebook
        - get_options
        - set_options
        - Options
        - DEFAULT_SLOT_ORDERS

## Common Utilities

::: xarray_plotly.common
    options:
      show_root_heading: true
      members:
        - auto
        - SLOT_ORDERS
        - assign_slots
