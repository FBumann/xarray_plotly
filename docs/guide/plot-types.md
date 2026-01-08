# Plot Types

xarray-plotly provides several plot types through the `pxplot` accessor.

## Line Plot

Create interactive line plots with `line()`:

```python
fig = da.pxplot.line()
```

**Slot order**: `x -> color -> line_dash -> symbol -> facet_col -> facet_row -> animation_frame`

**Special slots**:

- `line_dash`: Different line styles (solid, dashed, dotted)
- `symbol`: Different marker symbols when markers are enabled

```python
# With markers
fig = da.pxplot.line(markers=True)
```

## Bar Chart

Create interactive bar charts with `bar()`:

```python
fig = da.pxplot.bar()
```

**Slot order**: `x -> color -> pattern_shape -> facet_col -> facet_row -> animation_frame`

**Special slots**:

- `pattern_shape`: Different fill patterns (useful for black & white printing)

```python
# Grouped bars
fig = da.pxplot.bar(barmode="group")
```

## Area Chart

Create stacked area charts with `area()`:

```python
fig = da.pxplot.area()
```

**Slot order**: `x -> color -> pattern_shape -> facet_col -> facet_row -> animation_frame`

The `color` dimension determines the stacking order.

## Scatter Plot

Create scatter plots with `scatter()`:

```python
fig = da.pxplot.scatter()
```

**Slot order**: `x -> color -> size -> symbol -> facet_col -> facet_row -> animation_frame`

**Special features**:

- `y` can be set to a dimension name for dimension-vs-dimension plots
- `color` can be set to `"value"` to color by DataArray values

```python
# Geographic-style: lat vs lon, colored by value
da_geo = xr.DataArray(
    np.random.rand(10, 20),
    dims=["lat", "lon"],
    name="temperature",
)
fig = da_geo.pxplot.scatter(x="lon", y="lat", color="value")
```

## Box Plot

Create box plots with `box()`:

```python
fig = da.pxplot.box()
```

**Slot order**: `x -> color -> facet_col -> facet_row -> animation_frame`

!!! note
    By default, only `x` is auto-assigned. Other dimensions are aggregated into the box statistics. Use explicit assignment to show them:

```python
# Show both city (x) and scenario (color)
fig = da.pxplot.box(x="city", color="scenario")
```

## Heatmap (imshow)

Create heatmap images with `imshow()`:

```python
fig = da.pxplot.imshow()
```

**Slot order**: `y (rows) -> x (columns) -> facet_col -> animation_frame`

For heatmaps, both `x` and `y` are dimensions (not just `x` like other plots).

```python
da_2d = xr.DataArray(
    np.random.rand(10, 20),
    dims=["lat", "lon"],
)

# lat on y-axis (rows), lon on x-axis (columns)
fig = da_2d.pxplot.imshow()

# Swap axes
fig = da_2d.pxplot.imshow(x="lat", y="lon")
```

## Common Parameters

All plot methods accept these common parameters:

| Parameter | Description |
|-----------|-------------|
| `facet_col` | Dimension for subplot columns |
| `facet_row` | Dimension for subplot rows |
| `animation_frame` | Dimension for animation slider |
| `**px_kwargs` | Any Plotly Express argument |

### Faceting

Create subplot grids:

```python
fig = da.pxplot.line(facet_col="scenario", facet_row="region")
```

### Animation

Create animated plots:

```python
fig = da.pxplot.line(animation_frame="year")
```

### Passing Plotly Express Arguments

Any argument accepted by the underlying Plotly Express function can be passed:

```python
fig = da.pxplot.line(
    title="My Plot",
    color_discrete_sequence=["red", "blue"],
    template="plotly_white",
)
```
