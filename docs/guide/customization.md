# Customization

xarray-plotly returns standard Plotly `Figure` objects, giving you full access to Plotly's customization capabilities.

## Basic Customization

### Via Plot Arguments

Pass arguments directly to the plotting method:

```python
fig = da.pxplot.line(
    title="Temperature Over Time",
    color_discrete_sequence=["#1f77b4", "#ff7f0e", "#2ca02c"],
    template="plotly_white",
)
```

### Via update_layout

Modify the figure after creation:

```python
fig = da.pxplot.line()
fig.update_layout(
    title="Temperature Over Time",
    xaxis_title="Date",
    yaxis_title="Temperature (°C)",
    font_size=14,
)
```

## Labels from xarray Attributes

xarray-plotly automatically uses metadata from xarray attributes for labels:

```python
da = xr.DataArray(
    np.random.rand(10),
    dims=["time"],
    name="temperature",
    attrs={
        "long_name": "Air Temperature",
        "units": "K",
    },
)
da.coords["time"].attrs = {"long_name": "Time", "units": "hours"}

# Labels automatically show: "Air Temperature [K]" on y-axis
fig = da.pxplot.line()
```

Override with custom labels:

```python
fig = da.pxplot.line(
    labels={"temperature": "Temp (°C)", "time": "Hours"}
)
```

## Themes and Templates

Plotly has built-in templates:

```python
# Dark theme
fig = da.pxplot.line(template="plotly_dark")

# Minimal theme
fig = da.pxplot.line(template="simple_white")

# Seaborn-like
fig = da.pxplot.line(template="seaborn")
```

## Colors

### Discrete Colors

For categorical data (dimensions assigned to `color`):

```python
# Use a named color sequence
fig = da.pxplot.line(color_discrete_sequence=px.colors.qualitative.Set2)

# Use explicit colors
fig = da.pxplot.line(color_discrete_sequence=["red", "blue", "green"])
```

### Continuous Colors

For heatmaps and scatter plots with continuous color:

```python
# Use a named colorscale
fig = da.pxplot.imshow(color_continuous_scale="Viridis")

# Diverging colorscale centered at zero
fig = da.pxplot.imshow(
    color_continuous_scale="RdBu_r",
    color_continuous_midpoint=0,
)
```

## Facet Customization

Customize faceted plots:

```python
fig = da.pxplot.line(facet_col="city")

# Adjust facet spacing
fig.update_layout(
    margin=dict(l=50, r=50, t=80, b=50),
)

# Update all x-axes
fig.update_xaxes(tickangle=45)

# Update all y-axes
fig.update_yaxes(title_text="Value")
```

## Animation Customization

Customize animated plots:

```python
fig = da.pxplot.line(animation_frame="year")

# Adjust animation speed
fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 500
fig.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 300
```

## Exporting

### Interactive HTML

```python
fig.write_html("plot.html")
```

### Static Images

Requires `kaleido`:

```bash
pip install kaleido
```

```python
fig.write_image("plot.png", scale=2)  # 2x resolution
fig.write_image("plot.svg")
fig.write_image("plot.pdf")
```

## Advanced: Direct Trace Modification

Access and modify traces directly:

```python
fig = da.pxplot.line()

# Modify specific traces
for trace in fig.data:
    trace.line.width = 3

# Add a horizontal line
fig.add_hline(y=0, line_dash="dash", line_color="gray")

# Add annotations
fig.add_annotation(
    x=5,
    y=0.5,
    text="Important point",
    showarrow=True,
)
```

## Integration with Dash

xarray-plotly figures work seamlessly with Dash:

```python
from dash import Dash, dcc, html

app = Dash(__name__)

fig = da.pxplot.line()

app.layout = html.Div([
    dcc.Graph(figure=fig)
])
```
