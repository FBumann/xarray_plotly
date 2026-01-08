# xarray-plotly-accessor
Convenience plotting accessor for xarray

## Background: A PR to xarray which got rejected

### Is your feature request related to a problem?

The current `.plot` accessor in xarray is built on matplotlib, which has several limitations for modern data exploration workflows:

1. **Static outputs**: Matplotlib plots are non-interactive. Users cannot zoom, pan, hover for values, or toggle traces without significant additional code.

2. **Post-creation modification is cumbersome**: Customizing matplotlib plots after creation requires understanding the complex `Axes`/`Figure` object hierarchy. Common tasks like adjusting labels, colors, or adding annotations often require many lines of boilerplate.

3. **Multi-dimensional data visualization**: When datasets have 3+ dimensions, users must manually slice/aggregate before plotting. There's no built-in support for faceting, animation, or interactive dimension exploration.

4. **Maintenance burden**: The matplotlib plotting code in xarray is substantial and complex. A Plotly-based alternative could reduce this burden since Plotly's express API handles much of the layout logic internally.

5. **Missing modern visualization patterns**: Interactive heatmaps with hover info, animated time series, linked brushing between facets - these are increasingly expected in data science workflows but require significant custom code with matplotlib.

### Describe the solution you'd like

A new **`pxplot`** accessor (Plotly Express plot) that provides:

### Core API Design

```python
import xarray as xr

ds = xr.Dataset(...)
da = xr.DataArray(...)

# Basic usage - automatic dimension assignment
ds.pxplot.line()
ds.pxplot.bar()
ds.pxplot.area()
da.pxplot.heatmap()

# Explicit dimension-to-slot mapping
ds.pxplot.line(x='time', color='scenario', facet_col='region')

# Returns plotly.graph_objects.Figure for easy customization
fig = ds.pxplot.bar(color='variable')
fig.update_layout(title='My Custom Title')
fig.show()
```

### Automatic Dimension → Slot Assignment

Dimensions are assigned to plot "slots" **by their order** in the Dataset/DataArray:

| Slot | Purpose |
|------|---------|
| `x` | X-axis |
| `color` | Trace grouping/stacking |
| `facet_col` | Subplot columns |
| `facet_row` | Subplot rows |
| `animation_frame` | Animation slider |

**Default slot order** (per plot type):
```python
SLOT_ORDER = ('x', 'color', 'facet_col', 'facet_row', 'animation_frame')
```

**Assignment is purely positional** - no name-based heuristics:
```python
# Dataset with dims: ('time', 'scenario', 'region')
# Auto-assigns: time→x, scenario→color, region→facet_col
ds.pxplot.line()

# Dataset with dims: ('region', 'time', 'scenario')
# Auto-assigns: region→x, time→color, scenario→facet_col
ds.pxplot.line()
```

**Override modes for each slot:**
- `'dim_name'`: Explicitly use that dimension for this slot
- `None`: Skip this slot (don't assign any dimension to it)

```python
# Explicit assignment
ds.pxplot.line(x='time', color='scenario')

# Skip color slot: time→x, scenario→facet_col (color unused)
ds.pxplot.bar(x='time', color=None)
```

**Error on unassigned dimensions:**
If there are more dimensions than available slots, an error is raised. Users must reduce dimensionality first:
```python
# 6 dims but only 5 slots → Error
ds.pxplot.line()  # raises ValueError: 1 unassigned dimension(s): ['extra_dim'].
                  # Use .sel(), .isel(), or .mean() to reduce.

# Fix by reducing dimensions before plotting
ds.sel(extra_dim='value').pxplot.line()
ds.mean('extra_dim').pxplot.line()
```

### Handling Multi-Variable Datasets

For Datasets with multiple data variables, treat `'variable'` as a pseudo-dimension:

```python
ds = xr.Dataset({
    'temperature': (['time', 'station'], temp_data),
    'humidity': (['time', 'station'], humid_data),
})

# 'variable' can be assigned to color to compare temperature vs humidity
ds.pxplot.line(x='time', color='variable', facet_col='station')
```

### Proposed Methods

```python
# Dataset accessor
@xr.register_dataset_accessor('pxplot')
class DatasetPxplotAccessor:
    def line(self, *, x='auto', color='auto', facet_col='auto',
             facet_row='auto', animation_frame='auto', **px_kwargs) -> go.Figure

    def bar(self, *, x='auto', color='auto', ...) -> go.Figure

    def area(self, *, x='auto', color='auto', ...) -> go.Figure  # stacked area

    def scatter(self, *, x='auto', y='auto', color='auto', ...) -> go.Figure

    def heatmap(self, *, x='auto', y='auto', facet_col='auto', ...) -> go.Figure


# DataArray accessor
@xr.register_dataarray_accessor('pxplot')
class DataArrayPxplotAccessor:
    def line(self, ...) -> go.Figure
    def heatmap(self, ...) -> go.Figure
    # etc.
```

### Slot Order per Plot Type

Different plot types define their own default slot order:
```python
# line/bar/area: x is primary
LINE_SLOT_ORDER = ('x', 'color', 'facet_col', 'facet_row', 'animation_frame')

# heatmap: needs x and y for the grid
HEATMAP_SLOT_ORDER = ('x', 'y', 'facet_col', 'facet_row', 'animation_frame')

# scatter: x and y are primary
SCATTER_SLOT_ORDER = ('x', 'y', 'color', 'facet_col', 'facet_row', 'animation_frame')
```

Global configuration (e.g., custom slot orders, default colorscales) could be added later if needed.

### Example Usage

```python
import xarray as xr
import numpy as np

# Create sample dataset - dims order: (time, city, scenario)
ds = xr.Dataset({
    'temperature': (['time', 'city', 'scenario'], np.random.randn(100, 3, 2)),
    'precipitation': (['time', 'city', 'scenario'], np.random.randn(100, 3, 2)),
}, coords={
    'time': pd.date_range('2020', periods=100),
    'city': ['NYC', 'LA', 'Chicago'],
    'scenario': ['baseline', 'warming'],
})

# Dims: (time, city, scenario) + 'variable' (2 data vars)
# Slot order: (x, color, facet_col, facet_row)
# Result: time→x, city→color, scenario→facet_col, variable→facet_row
fig = ds.pxplot.line()

# Interactive: zoom, pan, hover for values, toggle traces
fig.show()

# Easy customization after creation
fig.update_layout(
    title='Climate Projections',
    xaxis_title='Date',
    template='plotly_dark',
)

# Override: put variable on color instead
# time→x, variable→color, city→facet_col, scenario→facet_row
fig = ds.pxplot.line(color='variable')

# Reduce dims first if you have too many
ds.sel(scenario='baseline').pxplot.line()  # 3 dims → fits in 3 slots
```

### Describe alternatives you've considered

### 1. External package (status quo)
Users can use `hvplot` (HoloViews-based) which provides a similar accessor. However:
- hvplot adds significant dependencies (HoloViews, Bokeh, Panel)
- Returns HoloViews objects, not Plotly figures
- Different ecosystem with its own learning curve

### 2. Improve matplotlib plotting
Could add faceting/animation to current `.plot`, but:
- Matplotlib's static nature is fundamental
- Would require major refactoring of existing code
- Doesn't address the post-creation modification pain point

### 3. Keep as third-party package
A `pxplot` package could exist independently. However:
- Discoverability suffers (users don't know it exists)
- Integration with xarray options/config is harder
- Fragmentation of the ecosystem

### Additional context

### Why Plotly Express specifically?

1. **Minimal API surface**: `px.line()`, `px.bar()`, etc. handle most layout concerns internally
2. **Returns modifiable objects**: `go.Figure` has a clean API for updates
3. **Wide adoption**: Plotly is well-known in the data science community
4. **Good defaults**: Sensible hover info, legends, and interactivity out of the box
5. **Export flexibility**: HTML, PNG, PDF, or embed in Dash/Jupyter

### Implementation Notes

The dimension→slot assignment algorithm is simple and predictable:

```python
SLOT_ORDER = ('x', 'color', 'facet_col', 'facet_row', 'animation_frame')

def assign_slots(ds, *, x=auto, color=auto, facet_col=auto, ...):
    """
    Positional assignment: dimensions fill slots in order.
    - Explicit assignments lock a dimension to a slot
    - None skips a slot
    - Remaining dims fill remaining slots by position
    - Error if dims left over after all slots filled
    """
    dims = list(ds.dims)
    if len(ds.data_vars) > 1:
        dims.append('variable')  # pseudo-dimension

    slots = {}
    used = set()
    slot_queue = list(SLOT_ORDER)

    # Pass 1: Process explicit assignments
    for slot, value in [('x', x), ('color', color), ...]:
        if value is None:
            slot_queue.remove(slot)  # skip this slot
        elif value is not auto:
            slots[slot] = value
            used.add(value)
            slot_queue.remove(slot)

    # Pass 2: Fill remaining slots with remaining dims (by position)
    remaining_dims = [d for d in dims if d not in used]
    for slot, dim in zip(slot_queue, remaining_dims):
        slots[slot] = dim
        used.add(dim)

    # Check for unassigned dimensions
    unassigned = [d for d in dims if d not in used]
    if unassigned:
        raise ValueError(
            f"Unassigned dimension(s): {unassigned}. "
            "Reduce with .sel(), .isel(), or .mean() before plotting."
        )

    return slots
```

### Proof of Concept

I have implemented this pattern in a domain-specific package ([flixopt](https://github.com/flixOpt/flixopt)) as `.fxplot` and it has proven valuable for exploring multi-dimensional optimization results. The patterns are generic and would benefit the broader xarray community.

### Dependency Considerations

This would add `plotly` as an optional dependency:
```python
try:
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError:
    raise ImportError("pxplot requires plotly. Install with: pip install plotly")
```

## Summary

A `pxplot` accessor would provide:
- Interactive plots with zero additional code
- Simple, predictable positional assignment of dimensions to visual slots
- Explicit overrides and `None` to skip slots
- Easy post-creation customization via Plotly's `go.Figure` API
- Reduced maintenance burden compared to matplotlib's complexity
- Modern visualization patterns (faceting, animation) built-in

This aligns with xarray's philosophy of making multi-dimensional data easy to work with, extending that ease to visualization.
