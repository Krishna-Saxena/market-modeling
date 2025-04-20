import numpy as np
import pandas as pd

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

def apply_one_encode(row, var_name, zeros_cols, level_id_map):
  if type(row[var_name]) == str or not np.isnan(row[var_name]):
    row[zeros_cols[level_id_map[row[var_name]]]] = 1
  return row

def apply_many_encode(row, sep, var_name, zeros_cols, level_id_map):
  if type(row[var_name]) == str and not np.isnan(row[var_name]):
    for value in row[var_name].split(sep):
      row[zeros_cols[level_id_map[value]]] = 1
  return row

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

def alpha_to_days(alpha):
  return int(2/alpha - 1)

def days_to_alpha(days):
  return 2/(days + 1)

def compute_next_ewma(EWMA_t, S_tdt, dt, alpha):
  """
  Compute next EWMA of a time series:
  EWMA_{t+dt} = alpha**dt * EWMA_t + (1 - alpha**dt) * S_{t+dt}.

  Args:
    EWMA_t: the EWMA up to time t.
    S_tdt: the state at time t + dt.
    dt: change in time between previous EWMA and new measurement.
    alpha: decay rate, 2/(Days + 1) is the common use case.

  Returns: EWMA_tdt, the next EWMA.
  """
  return alpha**dt * EWMA_t + (1-alpha**dt) * S_tdt

def make_ewma_col_name(var_name, days):
  return '{}_ewma_{}d'.format(var_name, int(round(days)))

def make_scale_diff_col_name(var_name):
  return '{}_scl_diff'.format(var_name)

def make_scale_diff_log_col_name(var_name):
  return '{}_log_scl_diff'.format(var_name)

def drop_from_array(array, values_to_drop):
  """
  Drops elements equal to the elements of `values_to_drop` from `array`.

  Args:
    array: a (not necessarily arithmetically, but always strictly) increasing array.
    values_to_drop: a (not necessarily arithmetically, but always strictly) increasing array.

  Returns: a subset of `array` that is still strictly increasing, without any elements in `values_to_drop`.
  """
  for i in range(len(array))[::-1]:
    if array[i] == values_to_drop[-1]:
      del array[i]
      values_to_drop.pop()
      if len(values_to_drop) == 0:
        break
  return array