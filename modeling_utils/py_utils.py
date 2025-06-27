from functools import lru_cache

import numpy as np
import pandas as pd

SEED = 802073711


########################## String Operations ##########################
def fmt_name(first, last):
	"""
	Returns: "{first} {last}".
	"""
	first = '' if not first else first
	last = '' if not last else last
	return f'{first} {last}'

def make_factor_col_name(var_name, level):
	"""
	Returns: "{var_name}_{level}".
	"""
	return f'{var_name}_{level}'

@lru_cache(5)
def make_ewma_col_name(var_name, days):
	"""
	Returns: "{var_name}_ewma_{days}d".
	"""
	return '{}_ewma_{}d'.format(var_name, int(round(days)))


@lru_cache(5)
def make_scale_diff_col_name(var_name):
	"""
	Returns: "{var_name}_scl_diff".
	"""
	return '{}_scl_diff'.format(var_name)


@lru_cache(5)
def make_scale_diff_log_col_name(var_name):
	"""
	Returns: "{var_name}_log_scl_diff".
	"""
	return '{}_log_scl_diff'.format(var_name)


########################## Array Operations ##########################
def last_index_lt_1D(a, value):
	"""

	Args:
		a: a 1D numpy array.
		value: a value satisfying `type(value) == a.dtype`.

	Returns: the last index `i` : `a[i] < value`.
	"""
	# https://stackoverflow.com/a/42945171
	idx = (a < value)[::-1].argmax()
	return a.shape[0] - idx - 1


def drop_values_from_array(array, values_to_drop):
	"""
	Drops elements equal to the elements of `values_to_drop` from `array`.

	Args:
		array: a (not necessarily arithmetically, but always strictly) increasing array.
		values_to_drop: a (not necessarily arithmetically, but always strictly) increasing array.

	Returns: a subset of `array` that is still strictly increasing, without any elements in `values_to_drop`.
	"""
	for i in range(len(array))[::-1]:
		if array[i] == values_to_drop[-1]:
			array = array[:i] + array[i+1:]
			values_to_drop.pop()
			if len(values_to_drop) == 0:
				break
	return array

def drop_indices_from_array(array, indices_to_drop):
	"""
	Drops elements equal to the elements of `values_to_drop` from `array`.

	Args:
		array: a (not necessarily arithmetically, but always strictly) increasing array.
		indices_to_drop: a (not necessarily arithmetically, but always strictly) increasing array.

	Returns: a subset of `array` that is still strictly increasing, without any elements in `values_to_drop`.
	"""
	for i in range(len(array))[::-1]:
		if i == indices_to_drop[-1]:
			array = array[:i] + array[i+1:]
			indices_to_drop.pop()
			if len(indices_to_drop) == 0:
				break
	return array


def standardize(A, mu, sigma):
	return (A - mu) / sigma


def add_random_noise_to_Cov_mat(Sigma, adjustment=100):
	# 1. set mu = [-1, 1]
	mu = np.array([-1, 1])
	# 2. run one iter of K-means clustering on triu(Sigma) with seed mu
	cov_vals = Sigma[np.triu_indices_from(Sigma, k=1)]
	assignments = np.argmin((cov_vals[:, None] - mu[None, :]) ** 2, axis=1)
	# 3. get vars of each mode
	var0 = 0. if np.allclose(assignments, 1) else np.var(cov_vals[assignments == 0])
	var1 = 0. if np.allclose(assignments, 0) else np.var(cov_vals[assignments == 1])
	# 4. let sig2_hat = weighted avg of the modes' vars, weighted by members of each mode
	sig2_hat = (var0 * np.sum(assignments == 0) + var1 * np.sum(assignments == 1)) / (assignments.shape[0])
	# 5. add N(0, sig2_hat*I) noise to Sigma
	# rng = np.random.RandomState(SEED)
	rng = np.random.default_rng()
	noise_mat = rng.normal(0, sig2_hat ** 0.5 / adjustment, Sigma.shape).astype(np.float32)
	return Sigma + noise_mat@noise_mat.T

def iqr_clip(arr, S=3):
	"""
	Clips values of arr to the interval [median(a.flatten()) +/- S*IQR(a.flatten())]

	Args:
		arr: a np.array of any shape.
		S: scale factor.

	Returns: arr, but clipped.
	"""
	median = np.median(arr)
	iqr = np.percentile(arr, 75) - np.percentile(arr, 25)
	return np.clip(arr, median - S*iqr, median + S*iqr)

########################## Dataframe Operations ##########################
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
		df: a pd.DataFrame.
		cols_to_exclude: a list of column names whose values will not be filled. **All other columns will be filled**.

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
		df: a pd.DataFrame.
		cols_to_exclude: a list of column names whose values will not be filled. **All other columns will be filled**.

	Returns: df with all columns filled.
	"""
	covar_cols = df.columns.tolist()
	for col in cols_to_exclude:
		covar_cols.remove(col)
	df.iloc[:, covar_cols] = df.iloc[:, covar_cols].bfill()
	return df


########################## EWMA Operations ##########################
def alpha_to_days(alpha):
	"""
	Returns: int(2/alpha - 1).
	"""
	return int(2 / alpha - 1)


@lru_cache(5)
def days_to_alpha(days):
	"""
	Returns: 2/(days + 1).
	"""
	return 2 / (days + 1)


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
	return alpha ** dt * EWMA_t + (1 - alpha ** dt) * S_tdt
