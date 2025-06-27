from datetime import datetime
import pickle

import numpy as np
import pandas as pd
import xarray as xr

from metrics.EvalMetrics import mae
from modeling_utils.torch_utils import detach, make_Tensor, make_param_Tensor, get_dep_SRW_nll_loss, get_dep_Cov_SRW_nll_loss
from modeling_utils.xr_utils import append_xr_dataset, get_timestamps, get_dS_dt_timestamps, serve_xr_ds
from modeling_utils.calculus_utils import get_mle_dep_SRW_params, get_mle_dep_Cov_SRW_params


# Independent Model
def get_indep_MLE_params(growth_ts: xr.Dataset, col_to_model: str = 'signal'):
	"""
	Calculates the MLE RW parameters using a single timeseries.

	Args:
		`growth_ts`: a xr.Dataset with single index `time` and variables `signal` (plus other unused variables).
		`col_to_model`: a column in `ts_df` whose 1D GBM parameters will be returned.

	Returns: mu_hat, sigma_sq_hat, the MLE drift and volatility.
	"""
	if isinstance(growth_ts, xr.Dataset):
		dt = (growth_ts.time - growth_ts.time.shift(time=1)).dt.days.values
	elif isinstance(growth_ts, pd.DataFrame):
		dt = (growth_ts.time - growth_ts.time.shift(1)).dt.days.values
	signal = growth_ts[col_to_model].values

	delta_S = signal[-1] - signal[0]
	delta_t = np.nansum(dt)

	mu_hat = delta_S / delta_t
	sigma_sq_hat = np.nanmean((signal[1:] - signal[:-1] - mu_hat)**2 / dt[1:])

	return mu_hat, sigma_sq_hat


def sample_indep_arith(mu, sigma_sq, S_0, times, add_BM=True):
	"""
	Sample an independent Random Walk model.

	Args:
		mu: drift of GBM.
		sigma_sq: variance of GBM.
		S_0: initial signal.
		times: array of time steps since t_0.
		add_BM: whether to add Brownian Motion to sample.

	Returns: an array of sampled signals.
	"""
	vols = np.random.normal(size=times.shape)

	S_ts = np.zeros((len(times) + 1,))
	S_ts[0] = S_0
	for i in range(len(times)):
		dt = times[i] if i == 0 else times[i] - times[i - 1]
		vol = (sigma_sq * dt) ** 0.5 * vols[i - 1] if add_BM else 0
		S_ts[i + 1] = S_ts[i] + (mu * dt + vol)

	return S_ts


def ffill_1D_arith(ts_df, num_sims=100, col_to_fill='signal'):
	"""
	Forward fills NaN values of a single dataframe column with average simulations from a 1D RW model.

	Args:
		ts_df: a pd.DataFrame with columns `time` and `col_to_fill` (at minimum).
		num_sims: number of simulations for each NaN value, 100 by default.
		col_to_fill: a column in `ts_df` whose NaN values will be filled, 'signal' by default.

	Returns: `ts_df` with NaN values of `col_to_fill` filled. Note: `ts_df` was modified.
	"""
	full_df = ts_df.dropna(axis='index', subset=col_to_fill)
	# if no rows were dropped, return early before simulating
	if full_df.shape[-2] == ts_df.shape[-2]:
		return ts_df
	mu_hat, sigma_sq_hat = get_indep_MLE_params(full_df, col_to_fill)
	sigma_sq_hat /= 100.

	dt = (ts_df['time'] - ts_df['time'].shift(1)).dt.days.values
	sim_results = np.tile(ts_df[col_to_fill].values.reshape(-1, 1), (1, num_sims))
	for i in range(1, ts_df.shape[0]):
		if np.isnan(ts_df[col_to_fill].iloc[i]):
			sim_results[i, :] = sim_results[i - 1, :] + dt[i]*mu_hat + (dt[i]*sigma_sq_hat)**0.5*np.random.randn(num_sims)
	ts_df[col_to_fill] = np.mean(sim_results, axis=1)
	return ts_df


def bfill_1D_arith(ts_df, num_sims=100, col_to_fill='signal'):
	"""
	Backward fills NaN values of a single dataframe column with average simulations from a 1D RW model.

	Args:
		ts_df: a pd.DataFrame with columns `time` and `col_to_fill` (at minimum).
		num_sims: number of simulations for each NaN value, 100 by default.
		col_to_fill: a column in `ts_df` whose NaN values will be filled, 'signal' by default.

	Returns: `ts_df` with NaN values of `col_to_fill` filled. Note: `ts_df` was modified.
	"""
	full_df = ts_df.dropna(axis='index', subset=col_to_fill)
	# if no rows were dropped, return early before simulating
	if full_df.shape[-2] == ts_df.shape[-2]:
		return ts_df
	mu_hat, sigma_sq_hat = get_indep_MLE_params(full_df)
	sigma_sq_hat /= 100.

	dt = (ts_df['time'].shift(-1) - ts_df['time']).dt.days.values
	sim_results = np.tile(ts_df[col_to_fill].values.reshape(-1, 1), (1, num_sims))
	for i in range(ts_df.shape[0] - 1, 0, -1):
		if np.isnan(ts_df[col_to_fill].iloc[i-1]):
			sim_results[i-1, :] = sim_results[i, :] - dt[i-1]*mu_hat - (dt[i-1]*sigma_sq_hat)**0.5*np.random.randn(num_sims)
	ts_df[col_to_fill] = np.mean(sim_results, axis=1)
	return ts_df


# Dependent Model
def get_dep_MLE_params(
	train_ds: xr.Dataset, val_ds: xr.Dataset
):
	"""
	Calculate MLE MV-RW model of a vector of signals.

	Args:
		train_ds: a xr.Dataset with axes 'ID, 'time', 'variable' and 'variable' minimally containing 'signal'.
		val_ds: a xr.Dataset with axes 'ID, 'time', 'variable' and 'variable' minimally containing 'signal'.
		n_epochs: number of epochs of gradient ascent to run.
		steps_per_batch: number of training steps/batch/epoch.
		return_best: whether to return the best parameters (by MAE metric) or parameters from last iteration.

	Returns: (mu_hat, Sigma_hat), {'train_loss': [NLL], 'val_loss': [NLL], 'train_MAE': [MAE], 'val_MAE': [MAE]}.
	"""
	train_dS, train_dt, train_timestamps = get_dS_dt_timestamps(train_ds.to_dataarray())
	if val_ds is not None:
		val_dS, val_dt, val_timestamps = get_dS_dt_timestamps(val_ds.to_dataarray())


	mu_hat, Sigma_hat = get_mle_dep_SRW_params(train_dS, train_dt)

	return (mu_hat, Sigma_hat), {
		'epochs': [1],
		'train_loss': [get_dep_SRW_nll_loss(make_Tensor(train_dS), make_Tensor(train_dt), make_Tensor(mu_hat), make_Tensor(Sigma_hat)).detach().item()],
		'val_loss': [get_dep_SRW_nll_loss(make_Tensor(val_dS), make_Tensor(val_dt), make_Tensor(mu_hat), make_Tensor(Sigma_hat)).detach().item()] if val_ds is not None else [np.nan],
		'train_MAE': [
			mae(train_ds['signal'].to_numpy().T[:, 1:],
			    sample_dep_RW(mu_hat, Sigma_hat, train_ds['signal'].isel(time=0).to_numpy(), train_timestamps, False)[:, 1:])
		],
		'val_MAE': [
			mae(val_ds['signal'].to_numpy().T[:, 1:],
			    sample_dep_RW(mu_hat, Sigma_hat, val_ds['signal'].isel(time=0).to_numpy(), val_timestamps, False)[:, 1:])
		] if val_ds is not None else [np.nan]
	}


def sample_dep_RW(mu, Sigma, S_0, times, add_BM=True):
	"""
	Sample an independent Random Walk.

	Args:
		mu: GBM drift, shape (n_assets,).
		Sigma: the covariance matrix of Brownian Motions, shape (n_assets, n_assets)
		S_0: initial signal, shape (n_assets,).
		times: array of time steps since t_0.
		add_BM: whether to add Brownian Motion to sample, defaults to True.

	Returns: an array of [S_0 | sampled signals] with shape (S_0.shape[0], len(times) + 1).
	"""
	if add_BM:
		bm = np.random.normal(size=(S_0.shape[0], times.shape[0]))

	S_ts = np.zeros((S_0.shape[0], len(times) + 1))
	S_ts[:, 0] = S_0
	for i in range(len(times)):
		dt = times[i] if i == 0 else times[i] - times[i - 1]
		vol = (dt ** 0.5 * Sigma @ bm[:, i - 1]) if add_BM else 0
		S_ts[:, i + 1] = S_ts[:, i] + mu*dt + vol

	return S_ts


# Dependent Model w/ Covariates
def _get_weighted_MAE(dss, param_memos, theta, var_types, der_var_specs, covar_vars):
	_mae = 0.
	cum_weight = 0
	for ds, param_memo in zip(dss, param_memos):
		xarr = ds.to_dataarray()
		mu, Sigma = detach(param_memo[0]), detach(param_memo[1])
		train_sim = sample_dep_cov_RW(
			mu, Sigma, theta,
			xarr.sel(variable='signal').isel(time=0).to_numpy(), get_timestamps(xarr), ds, 0,
			var_types, der_var_specs, covar_vars,
			False
		)
		_mae += mae(xarr.sel(variable='signal').to_numpy().T[:, 1:], train_sim[:, 1:]) * xarr.shape[0] * xarr.shape[1]
		cum_weight += xarr.shape[0] * xarr.shape[1]
	return _mae / cum_weight

def get_dep_cov_MLE_params(
	covar_vars, train_dss: [xr.Dataset],
	**kwargs
):
	"""
	Calculate MLE MV-GBM model of a set of signals.

	Args:
		covar_vars: [str] A required list of covariate variable names, each entry `covar_vars[i]` must be a xr.Variable in `market.xarray_ds`.
		train_dss: [xr.Dataset] A list of xarray datasets containing the training data.
		val_dss: [xr.Dataset] A list of xarray datasets containing the validation data.

		**kwargs:
		- any kwargs supported by `serve_xr_ds()`.

	Returns: mu_hat, Sigma_hat, theta_hat.
	"""
	n_covars = len(covar_vars)
	mu, Sigma = None, None
	theta = np.zeros((n_covars, ))

	# average the theta's that optimize each dataset
	total_weight = 0
	for i, train_ds in enumerate(train_dss):
		# end shape: (n_assets, n_ts, n_covars)
		train_xarr = train_ds.to_dataarray()
		ds_covs = train_ds[covar_vars].to_dataarray().to_numpy().astype(np.float32).transpose((2, 1, 0))[:, :-1, :]
		dS, dt, timestamps = get_dS_dt_timestamps(train_xarr)
		mu, Sigma, theta_ds = get_mle_dep_Cov_SRW_params(dS, dt, ds_covs)

		weight = ds_covs.shape[0] * ds_covs.shape[1]
		theta += theta_ds*weight
		total_weight += weight
	theta = theta / total_weight

	return (mu, Sigma, theta), {
		'train_loss': [np.nan],
		'val_loss':[np.nan],
		'train_MAE':[np.nan],
		'val_MAE': [np.nan],
		'epochs':[1]
	}


def fit_dep_cov(
	covar_vars, train_ds: xr.Dataset,
	**kwargs
):
	"""
	Calculate MLE MV-GBM model of a set of signals.

	Args:
		covar_vars: [str] A required list of covariate variable names, each entry `covar_vars[i]` must be a xr.Variable in `market.xarray_ds`.
		train_ds: xr.Dataset A xarray dataset of training data.

		**kwargs:
		- any kwargs supported by `serve_xr_ds()`.

	Returns: mu_hat, Sigma_hat.
	"""
	train_xarr = train_ds.to_dataarray()
	ds_covs = train_ds[covar_vars].to_dataarray().to_numpy().astype(np.float32).transpose((2, 1, 0))[:, :-1, :]
	dS, dt, timestamps = get_dS_dt_timestamps(train_xarr)

	return get_mle_dep_SRW_params(dS - np.einsum('atc,c->at', ds_covs, kwargs.pop('theta')), dt)


def sample_dep_cov_RW(
	mu, Sigma, theta,
	S_0, times, ds, t_start_idx,
	var_types, der_var_specs, covar_vars,
	add_BM=True
):
	"""
	Sample an independent Brownian Motion.

	Args:
		mu: GBM drift, shape (n_assets,).
		Sigma: the covariance matrix of Brownian Motions, shape (n_assets, n_assets).
		theta: weights for covariates, shape (n_covars,).
		S_0: initial signal, shape (n_assets,).
		times: array of time steps since t_0.
		ds: xarray.Dataset of full market history, shape (n_assets, n_ts, n_covars).
		t_start_idx: index of `ds.sel(variable=COVAR_VARS)` corresponding to covars at t_0, will be mapped onto [0, len(ds['time'])-1). Generally, set t_start_idx to 0 to test GoF and t_start_idx to -1 to forecast.
		var_types: Dict[str, VarType], the types of variables in ds.
		der_var_specs: a list of DerivedVariable specifications.
		covar_vars: [str], A required list of covariate variable names, each entry `covar_vars[i]` must be a xr.Variable in `market.xarray_ds`.
		add_BM: whether to add Brownian Motion when generating samples.

	Returns: an array of [S_0 | sampled signals] with shape (S_0.shape[0], len(times) + 1).
	"""
	t_start_idx = 0 if len(ds['time']) == 1 else (t_start_idx + len(ds['time']) - 1) % (len(ds['time']) - 1)

	if add_BM:
		bm = np.random.normal(size=(S_0.shape[0], times.shape[0]))

	S_ts = np.zeros((S_0.shape[0], len(times) + 1))
	S_ts[:, 0] = S_0
	for i in range(len(times)):
		dt = times[i] if i == 0 else times[i] - times[i - 1]
		vol = (dt ** 0.5 * Sigma @ bm[:, i - 1]) if add_BM else 0
		covars = ds[covar_vars].isel(time=t_start_idx + i).to_dataarray().astype(np.float32).to_numpy()
		covar_contr = np.einsum('Ca,C->a', covars, theta)
		S_ts[:, i + 1] = S_ts[:, i] + (mu + covar_contr)*dt + vol
		if t_start_idx + i >= len(ds['time']):
			ds = append_xr_dataset(
				ds,
				S_ts[:, i + 1],
				dt,
				var_types,
				der_var_specs
			)

	return S_ts

