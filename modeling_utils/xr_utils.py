import xarray as xr
import xarray.core.dataset
import numpy as np

def extrapolate_covar_xr(
  ds: xarray.core.dataset.Dataset,
  new_S,
  dt: int,
  avg_map: dict=dict()
):
  # 1. extend time axis
  last_frame = ds.isel({'time':-1}).copy()
  last_frame['time'] = last_frame['time'] + np.timedelta64(dt, 'D')
  ds = xr.concat((ds, last_frame), dim='time')
  # 2. replace last signal
  ds.isel({'time':-1})['signal'].values = new_S
  # 3. recompute EWMAs
  for var_name, time_spans in avg_map.items():
    for time_span in time_spans:
      avg_var_name = f'{var_name}_{time_span}d_avg'
      ds[avg_var_name] = ds.sel(variable=avg_var_name).rolling(time_span).mean()
  return ds