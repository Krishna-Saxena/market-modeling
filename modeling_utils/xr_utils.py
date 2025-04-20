import xarray as xr
import numpy as np
from xarray import Dataset, DataArray, Variable, zeros_like

from modeling_utils.py_utils import drop_from_array
from markets.DerivedVariables import DerivedVariable

def zeros_like_nan(arr):
  if isinstance(arr, DataArray) or isinstance(arr, Dataset) or isinstance(arr, Variable):
    new_arr = zeros_like(arr)
    if arr.dtype == object or arr.dtype == str:
      new_arr = new_arr.where(arr != '')
    else:
      new_arr = new_arr.where(arr != np.NaN)
  else:
    new_arr = np.zeros_like(arr)
    if arr.dtype == object or arr.dtype == str:
      new_arr[arr == ''] = np.NaN
    else:
      new_arr[np.isnan(arr)] = np.NaN
  return new_arr

def get_dataset_only_numeric_vars(dataset: Dataset):
  dataset_df = dataset.to_dataframe()

  for var_name, var_type in dataset_df.dtypes.items():
    if var_type == object:
      dataset_df.drop(var_name, axis=1, inplace=True)

  return Dataset.from_dataframe(dataset_df)

def ffill_via_pandas(ds: xr.Dataset) -> xr.Dataset:
  df_ID_t_indices = ds.to_dataframe()
  ffilled_df = df_ID_t_indices.groupby('ID', sort=False).ffill()
  return xr.Dataset.from_dataframe(ffilled_df)

def bfill_via_pandas(ds: xr.Dataset) -> xr.Dataset:
  df_ID_t_indices = ds.to_dataframe()
  ffilled_df = df_ID_t_indices.groupby('ID', sort=False).bfill()
  return xr.Dataset.from_dataframe(ffilled_df)

def append_xr_dataset(ds: xr.Dataset, new_S, dt: int, der_var_specs: [DerivedVariable]) -> xr.Dataset:
  """
  Append a xr.Dataset `ds` by copying the oldest values for covariates, setting S_{t+dt} to `new_S`, and recalculating the covariates using `der_var_specs`.

  Args:
    ds: a xr.Dataset.
    new_S: an array of shape (N_ASSETS,).
    dt: the number of days since the final timestep in `ds` that `new_S` was sampled at.
    der_var_specs: a list of DerivedVariable specifications.

  Returns: a new xr.Dataset that has the same structure as `ds`, but one more time value.
  """
  # 1. extend time axis
  last_frame = ds.isel({'time':-1}).copy()
  last_frame['time'] = last_frame['time'] + np.timedelta64(dt, 'D')
  ds = xr.concat((ds, last_frame), dim='time')
  # 2. replace last signal
  ds['signal'].data[-1, :] = new_S
  # 3. recompute data-driven covars
  for der_var_spec in der_var_specs:
    ds = der_var_spec.update_var(ds, dt=dt)
  return ds

def get_var_names(ds: xr.Dataset):
  return [k for k, v in ds.items()]

def get_covar_indices_and_names(ds: xr.Dataset, covar_codes):
  # find the indices of each requested covariate
  covar_indices, covar_names = [], []
  covar_var_set = set(covar_codes)
  var_names = get_var_names(ds)
  for col_i, col_name in enumerate(var_names):
    if col_name in covar_var_set:
      covar_indices.append(col_i)
      covar_names.append(col_name)
  return covar_indices, covar_names


SUPPORTED_SHUFFLE_TYPES = {
  'none',
  'shuffle',
  'mix'
}

def serve_xr_ds(ds: xr.Dataset, batch_size=None, shuffle_mode='none'):
  """
  Serve a xr.Dataset in a Python generator. `ds` can be batched along the time axis.
  Supports three shuffle schemes as described below.

  Args:
    `ds`: a xr.Dataset with axes 'ID', 'time'.
    `batch_size`: number of samples in each batch, must be >= 4 but suggested to be >= 20. If `batch_size` is None or >= the number of entries in `ds`, yields `ds`.
    `shuffle_mode`: one of three values:
    - 'none': yields chronological batches of size `batch_size`.
    - 'shuffle': yields randomly shuffled, chronologically contiguous, batches of size `batch_size`.
    - 'mix': yields randomly shuffled, chronologically increasing but stochastically discontiguous, batches of size `batch_size`.

  Returns: batches of `ds`
  """
  num_timesteps = ds['time'].values.shape[0]
  num_batches = int(num_timesteps / batch_size)

  assert batch_size is None or (isinstance(batch_size, int) and batch_size >= 4), '`batch_size` must be None or an integer >= 4.'
  if batch_size is None or batch_size >= num_timesteps:
    yield ds
    raise StopIteration
  elif shuffle_mode == 'none':
    for i in range(num_batches):
      yield ds.isel(time=slice(i*batch_size, min((i+1)*batch_size, num_timesteps)))
    raise StopIteration
  # end `elif shuffle_mode == 'none'`
  elif shuffle_mode == 'shuffle':
    order = np.random.permutation(num_batches)
    for i in range(num_batches):
      yield ds.isel(time=slice(order[i]*batch_size, min((order[i]+1)*batch_size, num_timesteps)))
    raise StopIteration
  # end `elif shuffle_mode == 'shuffle'`
  elif shuffle_mode == 'mix':
    indices, indices_st, batch_indices = np.arange(num_timesteps), 0, []
    coin_flips = np.random.binomial(1, 1-1/batch_size, num_timesteps).tolist()

    while (len(indices) > batch_size) and (len(batch_indices) < batch_size):
      # w/ prob = 1 - 1/batch_size, add the first unseen, unadded index to the batch
      if coin_flips[-1] == 1:
        batch_indices.append(indices[indices_st])
      indices_st += 1
      coin_flips.pop()
      # yield a full batch if we have one
      if len(batch_indices) == batch_size:
        yield ds.isel(time=batch_indices)
        # remove the yielded data's indices from future consideration
        indices, indices_st, batch_indices = drop_from_array(indices, batch_indices), 0, []
    # yield leftover indices iff there are enough for a full batch
    if len(indices) + len(batch_indices) >= batch_size:
      leftover_indices = sorted([*indices_st, *batch_indices])
      yield ds.isel(time=leftover_indices)
    raise StopIteration
  # end `elif shuffle_mode == 'mix'`