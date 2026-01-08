# Dimension Assignment

xarray-plotly uses a simple, predictable algorithm to assign dimensions to plot slots.

## Slot Order

Dimensions are assigned to "slots" based on their position in the DataArray. Each plot type has its own slot order:

### Line, Bar, Area Plots

```
x -> color -> line_dash/pattern_shape -> symbol -> facet_col -> facet_row -> animation_frame
```

### Scatter Plots

```
x -> color -> size -> symbol -> facet_col -> facet_row -> animation_frame
```

### Heatmap (imshow)

```
y (rows) -> x (columns) -> facet_col -> animation_frame
```

### Box Plots

```
x -> color -> facet_col -> facet_row -> animation_frame
```

!!! note
    For box plots, only `x` is auto-assigned by default. Other dimensions are aggregated into the box statistics unless explicitly assigned.

## Automatic Assignment

Dimensions fill slots in the order they appear in the DataArray:

```python
import xarray as xr
import numpy as np
import xarray_plotly

da = xr.DataArray(
    np.random.rand(10, 3, 2),
    dims=["time", "city", "scenario"],
)

# Dims: (time, city, scenario)
# Slots: (x, color, facet_col, ...)
# Result: time->x, city->color, scenario->facet_col
fig = da.pxplot.line()
```

If you reorder the dimensions, the assignment changes:

```python
da_reordered = da.transpose("city", "time", "scenario")

# Dims: (city, time, scenario)
# Result: city->x, time->color, scenario->facet_col
fig = da_reordered.pxplot.line()
```

## Explicit Assignment

Override automatic assignment by specifying dimensions explicitly:

```python
# Put scenario on color instead of city
fig = da.pxplot.line(color="scenario")
# Result: time->x, scenario->color, city->facet_col

# Fully explicit assignment
fig = da.pxplot.line(x="city", color="time", facet_col="scenario")
```

## Skipping Slots

Use `None` to skip a slot:

```python
# Skip the color slot
fig = da.pxplot.line(color=None)
# Result: time->x, city->facet_col, scenario->facet_row
```

## Too Many Dimensions

If you have more dimensions than available slots, an error is raised:

```python
da_6d = xr.DataArray(np.random.rand(2, 2, 2, 2, 2, 2), dims=list("abcdef"))

# Error: Only 5 slots available for line plots after x
da_6d.pxplot.line()
# ValueError: Unassigned dimension(s): ['f']. Reduce with .sel(), .isel(), or .mean() before plotting.
```

Reduce dimensions before plotting:

```python
# Select a single value
da_6d.sel(f=0).pxplot.line()

# Or average over a dimension
da_6d.mean("f").pxplot.line()
```
