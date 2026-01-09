"""Partial stubs extending xarray.DataArray with plotly accessor."""

from xarray import (
    ALL_DIMS as ALL_DIMS,
)
from xarray import (
    Coordinates as Coordinates,
)
from xarray import (
    Dataset as Dataset,
)
from xarray import (
    DataTree as DataTree,
)
from xarray import (
    IndexVariable as IndexVariable,
)
from xarray import (
    Variable as Variable,
)
from xarray import (
    apply_ufunc as apply_ufunc,
)
from xarray import (
    broadcast as broadcast,
)
from xarray import (
    cftime_range as cftime_range,
)
from xarray import (
    concat as concat,
)
from xarray import (
    conventions as conventions,
)
from xarray import (
    date_range as date_range,
)
from xarray import (
    decode_cf as decode_cf,
)
from xarray import (
    dot as dot,
)
from xarray import (
    full_like as full_like,
)
from xarray import (
    group_subtrees as group_subtrees,
)
from xarray import (
    infer_freq as infer_freq,
)
from xarray import (
    load_dataarray as load_dataarray,
)
from xarray import (
    load_dataset as load_dataset,
)
from xarray import (
    map_blocks as map_blocks,
)
from xarray import (
    merge as merge,
)
from xarray import (
    ones_like as ones_like,
)
from xarray import (
    open_dataarray as open_dataarray,
)
from xarray import (
    open_dataset as open_dataset,
)
from xarray import (
    open_datatree as open_datatree,
)
from xarray import (
    open_groups as open_groups,
)
from xarray import (
    open_mfdataset as open_mfdataset,
)
from xarray import (
    open_zarr as open_zarr,
)
from xarray import (
    polyval as polyval,
)
from xarray import (
    register_dataarray_accessor as register_dataarray_accessor,
)
from xarray import (
    register_dataset_accessor as register_dataset_accessor,
)
from xarray import (
    save_mfdataset as save_mfdataset,
)
from xarray import (
    set_options as set_options,
)
from xarray import (
    show_versions as show_versions,
)
from xarray import (
    testing as testing,
)
from xarray import (
    tutorial as tutorial,
)
from xarray import (
    unify_chunks as unify_chunks,
)
from xarray import (
    where as where,
)
from xarray import (
    zeros_like as zeros_like,
)
from xarray.core.dataarray import DataArray as _OriginalDataArray

from xarray_plotly.accessor import DataArrayPlotlyAccessor

# Extend DataArray with the plotly accessor
class DataArray(_OriginalDataArray):
    @property
    def plotly(self) -> DataArrayPlotlyAccessor: ...
