import xarray as xr
import numpy as np

from modeling_utils.py_utils import make_avg_col_name, compute_next_ewma

def ffill_via_pandas(ds: xr.Dataset) -> xr.Dataset:
  df_ID_t_indices = ds.to_dataframe()
  ffilled_df = df_ID_t_indices.groupby('ID', sort=False).ffill()
  return xr.Dataset.from_dataframe(ffilled_df)

def bfill_via_pandas(ds: xr.Dataset) -> xr.Dataset:
  df_ID_t_indices = ds.to_dataframe()
  ffilled_df = df_ID_t_indices.groupby('ID', sort=False).bfill()
  return xr.Dataset.from_dataframe(ffilled_df)

def extrapolate_covar_xr(
  ds: xr.Dataset,
  new_S,
  dt: int,
  avg_map: dict=dict()
) -> xr.Dataset:
  # 1. extend time axis
  last_frame = ds.isel({'time':-1}).copy()
  last_frame['time'] = last_frame['time'] + np.timedelta64(dt, 'D')
  ds = xr.concat((ds, last_frame), dim='time')
  # 2. replace last signal
  ds['signal'].data[-1, :] = new_S
  # 3. recompute EWMAs
  # for var_name, time_spans in avg_map.items():
  var_name = 'signal'
  for time_span in avg_map.get('signal', []):
    alpha = 2./(time_span + 1)
    avg_var_name = make_avg_col_name(var_name, time_span)
    ds[avg_var_name].data[-1, :] = compute_next_ewma(
      ds.sel(variable=avg_var_name).isel({'time':-1}),
      ds.sel(variable=var_name).isel({'time': -1}),
      dt,
      alpha
    )
  return ds