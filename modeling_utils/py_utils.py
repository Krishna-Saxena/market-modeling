import numpy as np
import pandas as pd
from xarray import Dataset, DataArray, Variable, zeros_like


def last_index_lt_1D(a, value):
  """

  Args:
    a: a 1D numpy array
    value: a value satisfying type(value) == a.dtype

  Returns: the last index i : a[i] < value

  """
  # https://stackoverflow.com/a/42945171
  idx = (a < value)[::-1].argmax()
  return a.shape[0] - idx - 1

def fmt_name(first, last):
  first = '' if not first else first
  last = '' if not last else last
  return f'{first} {last}'

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

def apply_one_encode(row, var_name, zeros_cols, level_id_map):
  if type(row[var_name]) == str or not np.isnan(row[var_name]):
    row[zeros_cols[level_id_map[row[var_name]]]] = 1
  return row

def apply_many_encode(row, sep, var_name, zeros_cols, level_id_map):
  if type(row[var_name]) == str and not np.isnan(row[var_name]):
    for value in row[var_name].split(sep):
      row[zeros_cols[level_id_map[value]]] = 1
  return row

def get_dataset_only_numeric_vars(dataset: Dataset):
  dataset_df = dataset.to_dataframe()

  for var_name, var_type in dataset_df.dtypes.items():
    if var_type == object:
      dataset_df.drop(var_name, axis=1, inplace=True)

  return Dataset.from_dataframe(dataset_df)

def make_avg_col_name(var_name, time_span):
  return f'{var_name}_{time_span}d_avg'

def ffill_df(df: pd.DataFrame, cols_to_exclude):
  """
  Forward fill a subset of columns in a DataFrame.

  Args:
    df: a dataframe
    cols_to_exclude: a list of column names whose values will not be filled. **All other columns will be filled.**

  Returns: df with all columns filled.
  """
  covar_cols = df.columns.tolist()
  for col in cols_to_exclude:
    covar_cols.remove(col)
  df.iloc[:, covar_cols] = df.iloc[:, covar_cols].ffill()
  return df

def bfill_df(df: pd.DataFrame, cols_to_exclude):
  """
  Backward fill a subset of columns in a DataFrame.

  Args:
    df: a dataframe
    cols_to_exclude: a list of column names whose values will not be filled. **All other columns will be filled.**

  Returns: df with all columns filled.
  """
  covar_cols = df.columns.tolist()
  for col in cols_to_exclude:
    covar_cols.remove(col)
  df.iloc[:, covar_cols] = df.iloc[:, covar_cols].bfill()
  return df