from datetime import datetime
import pickle
import warnings

import numpy as np
import pandas as pd
import xarray as xr
import torch
from tqdm import trange

from metrics.EvalMetrics import mae
from modeling_utils.py_utils import add_random_noise_to_Cov_mat
from modeling_utils.torch_utils import get_zero_grad_hook, diag_mask_gen_fn, triu_mask_gen_fn, detach, make_Tensor, make_param_Tensor, get_dep_SRW_nll_loss, get_dep_Cov_SRW_nll_loss
from modeling_utils.xr_utils import append_xr_dataset, get_timestamps, get_dX_dt_timestamps, serve_xr_ds
from modeling_utils.calculus_utils import get_mle_dep_SRW_params, get_mle_dep_Cov_SRW_params


# Independent Model
def get_indep_MLE_params(growth_ts: xr.Dataset, col_to_model: str = 'signal'):
	"""
	Calculates the MLE GBM parameters using a single timeseries.

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

	n = signal.shape[0]
	log_prices = np.log(signal)
	delta_X = log_prices[-1] - log_prices[0]
	delta_t = np.nansum(dt)

	sigma_sq_hat = -1 / n * delta_X ** 2 / delta_t + np.nanmean((log_prices[1:] - log_prices[:-1]) ** 2 / dt[1:])
	mu_hat = delta_X / delta_t + 0.5 * sigma_sq_hat

	return mu_hat, sigma_sq_hat


def sample_indep_GBM(mu, sigma_sq, S_0, times, add_BM=True):
	"""
	Sample an independent Geometric Brownian Motion.

	Args:
		mu: drift of GBM.
		sigma_sq: variance of GBM.
		S_0: initial signal.
		times: array of time steps since t_0.
		add_BM: whether to add Brownian Motion to sample.

	Returns: an array of sampled signals.
	"""
	avg_drift = mu - sigma_sq / 2
	vols = np.random.normal(size=times.shape)

	S_ts = np.zeros((len(times) + 1,))
	S_ts[0] = S_0
	for i in range(len(times)):
		dt = times[i] if i == 0 else times[i] - times[i - 1]
		vol = (sigma_sq * dt) ** 0.5 * vols[i - 1] if add_BM else 0
		S_ts[i + 1] = S_ts[i] * np.exp(avg_drift * dt + vol)

	return S_ts


def ffill_1D_GBM(ts_df, num_sims=100, col_to_fill='signal'):
	"""
	Forward fills NaN values of a single dataframe column with average simulations from a 1D GBM model.

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
	mu_hat /= (num_sims**0.5)
	sigma_sq_hat /= num_sims
	avg_drift = mu_hat - sigma_sq_hat / 2

	dt = (ts_df['time'] - ts_df['time'].shift(1)).dt.days.values
	sim_results = np.tile(ts_df[col_to_fill].values.reshape(-1, 1), (1, num_sims))
	for i in range(1, ts_df.shape[0]):
		if np.isnan(ts_df[col_to_fill].iloc[i]):
			sim_results[i, :] = sim_results[i - 1, :] * np.exp(
				dt[i] * avg_drift + (dt[i] * sigma_sq_hat) ** 0.5 * np.random.randn(num_sims))
	with warnings.catch_warnings():
		warnings.filterwarnings('ignore', category=RuntimeWarning)
		imputed_vals = np.nanmean(
			np.where(
				(sim_results>np.percentile(sim_results, 25, axis=1, keepdims=True))&(sim_results < np.percentile(sim_results, 75, axis=1, keepdims=True)),
				sim_results,
				np.nan),
			axis=1
		)
	ts_df[col_to_fill] = np.where(np.isnan(ts_df[col_to_fill]), imputed_vals, ts_df[col_to_fill])
	return ts_df


def bfill_1D_GBM(ts_df, num_sims=100, col_to_fill='signal'):
	"""
	Backward fills NaN values of a single dataframe column with average simulations from a 1D GBM model.

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
	mu_hat /= (num_sims ** 0.5)
	sigma_sq_hat /= num_sims
	avg_drift = mu_hat - sigma_sq_hat / 2

	dt = (ts_df['time'].shift(-1) - ts_df['time']).dt.days.values
	sim_results = np.tile(ts_df[col_to_fill].values.reshape(-1, 1), (1, num_sims))
	for i in range(ts_df.shape[0] - 1, 0, -1):
		if np.isnan(ts_df[col_to_fill].iloc[i - 1]):
			sim_results[i - 1, :] = sim_results[i, :] * np.exp(
				-dt[i - 1] * avg_drift - (dt[i - 1] * sigma_sq_hat) ** 0.5 * np.random.randn(num_sims))
	with warnings.catch_warnings():
		warnings.filterwarnings('ignore', category=RuntimeWarning)
		imputed_vals = np.nanmean(
			np.where(
				(sim_results>np.percentile(sim_results, 25, axis=1, keepdims=True))&(sim_results < np.percentile(sim_results, 75, axis=1, keepdims=True)),
				sim_results,
				np.nan),
			axis=1
		)
	ts_df[col_to_fill] = np.where(np.isnan(ts_df[col_to_fill]), imputed_vals, ts_df[col_to_fill])
	return ts_df


# Dependent Model
def _get_nll_loss_dep(dX, dt, mu, Sigma):
	drift = mu - torch.einsum('sS,sS->s', Sigma, Sigma) / 2
	return get_dep_SRW_nll_loss(dX, dt, drift, Sigma)


def _init_mu_A_from_numpy(mu_0, A_0, mask_gen_fn=triu_mask_gen_fn):
	mu = make_param_Tensor(mu_0)
	A = make_param_Tensor(A_0)
	# keep A as a lower triangular matrix to reduce feature space
	#   https://discuss.pytorch.org/t/creating-a-triangular-weight-matrix-with-a-hidden-layer-instead-of-a-fully-connected-one/71427/
	mask = mask_gen_fn(A)
	A.register_hook(get_zero_grad_hook(mask))
	return mu, A


def get_dep_MLE_params(
	train_ds: xr.Dataset, val_ds: xr.Dataset,
	n_epochs=50, steps_per_batch=10, lr=1e-4, lr_decay=0.1**0.5,
	val_freq=1, return_best=True,
	save_hist=False
) -> [(np.array, np.array), dict]:
	"""
	Calculate MLE MV-GBM model of a set of signals.

	Args:
		train_ds: a xr.Dataset with axes 'ID, 'time', 'variable' and 'variable' minimally containing 'signal'.
		val_ds: a xr.Dataset with axes 'ID, 'time', 'variable' and 'variable' minimally containing 'signal'.
		n_epochs: number of epochs of gradient ascent to run.
		steps_per_batch: number of training steps/batch/epoch.
		lr: learning rate.
		lr_decay: learning rate decay factor, applied every 5 epochs.
		val_freq: number of batches between validation metric calculations (excludes for the first 10 epochs when val metrics are always calculated).
		return_best: whether to return the best parameters (by MAE metric) or parameters from last iteration.
		save_hist: whether to save the training history in a pkl file.

	Returns: (mu_hat, Sigma_hat), {'train_loss': [NLL], 'val_loss': [NLL], 'train_MAE': [MAE], 'val_MAE': [MAE], 'mus': [np.array], 'Sigmas': [np.array], 'epochs': [int]}.
	"""
	train_xarr = train_ds.to_dataarray()
	train_dX, train_dt, train_timestamps = get_dX_dt_timestamps(train_xarr)
	if val_ds is not None:
		val_xarr = val_ds.to_dataarray()
		val_dX, val_dt, val_timestamps = get_dX_dt_timestamps(val_xarr)

	# get initial state from simple model
	mu_0, Sigma_0 = get_mle_dep_SRW_params(train_dX, train_dt)
	mu_0 -= np.einsum('sS,sS->s', Sigma_0, Sigma_0)/2
	Sigma_0 = add_random_noise_to_Cov_mat(Sigma_0)
	A_0 = np.linalg.cholesky(Sigma_0).real

	mu, A = _init_mu_A_from_numpy(mu_0, A_0)

	train_losses = []
	train_maes = []
	val_losses = []
	val_maes = []
	epochs = []
	mus = []
	Sigmas = []

	best_mu, best_Sigma = None, None
	best_metric = float('inf')

	for i in (p_bar := trange(n_epochs)):
		# reduce lr by a factor of 10 every 10 train epochs
		optim = torch.optim.Adam((mu, A), lr=lr*lr_decay**(i//5))

		for tr_batch in serve_xr_ds(train_ds, 20, 'shuffle'):
			# get the batch's signal, time data
			dX, dt, timestamps = get_dX_dt_timestamps(tr_batch.to_dataarray())
			dX = make_Tensor(dX)
			dt = make_Tensor(dt)
			# train model params on this batch
			for _ in range(steps_per_batch):
				optim.zero_grad()
				Sigma = A @ A.T
				loss = _get_nll_loss_dep(dX, dt, mu, Sigma)
				loss.backward()
				optim.step()

		train_dX_pt = make_Tensor(train_dX)
		train_dt_pt = make_Tensor(train_dt)
		if val_ds is not None:
			val_dX_pt = make_Tensor(val_dX)
			val_dt_pt = make_Tensor(val_dt)
		# logging, evaluation metrics
		with torch.no_grad():
			Sigma = A @ A.T
			epochs.append(i)
			train_loss = _get_nll_loss_dep(train_dX_pt, train_dt_pt, mu, Sigma).item()
			train_losses.append(train_loss)
			val_loss = _get_nll_loss_dep(val_dX_pt, val_dt_pt, mu, Sigma).item() if val_ds is not None else np.nan
			val_losses.append(val_loss)
			# save a history of the parameters, too
			mus.append(detach(mu))
			Sigmas.append(detach(Sigma))

			if (i < 10) or (i % val_freq == 0):
				#   train metrics
				train_sim = sample_dep_GBM(
					mus[-1], Sigmas[-1],
					train_xarr.sel(variable='signal').isel(time=0).to_numpy(),
					train_timestamps, False
				)
				# note [:, 1:] because the 1st col of sample_corr_GBM's result is S_0
				train_maes.append(mae(train_xarr.sel(variable='signal').to_numpy().T[:, 1:], train_sim[:, 1:]))
				if val_ds is not None:
					val_sim = sample_dep_GBM(
						mus[-1], Sigmas[-1],
						val_xarr.sel(variable='signal').isel(time=0).to_numpy(),
						val_timestamps, False
					)
					val_maes.append(mae(val_xarr.sel(variable='signal').to_numpy().T[:, 1:], val_sim[:, 1:]))
				else:
					val_maes.append(np.nan)

				p_bar.set_postfix({'train_loss': train_losses[-1], 'val_loss': val_losses[-1], 'train_MAE': train_maes[-1], 'val_MAE': val_maes[-1]})
				comp_metric = val_maes[-1] if val_ds is not None else train_maes[-1]
				if comp_metric < best_metric:
					best_metric = comp_metric
					best_mu = mus[-1]
					best_Sigma = Sigmas[-1]
			else:
				p_bar.set_postfix({'train_loss': train_losses[-1], 'val_loss': val_losses[-1]})

	train_hist_info = {
		'train_loss': train_losses,
		'val_loss': val_losses,
		'train_MAE': train_maes,
		'val_MAE': val_maes,
		'mus': mus,
		'Sigmas': Sigmas,
		'epochs': epochs
	}

	if save_hist:
		now_dt = datetime.now()
		with open(f'training_hist_dep_GBM_{datetime.strftime(now_dt, "%Y_%m_%d_%H_%M_%S")}.pkl', 'wb') as file:
			pickle.dump(train_hist_info, file)
		training_df = pd.DataFrame({'epochs': epochs, 'train_loss': train_losses, 'val_loss': val_losses, 'train_mae': train_maes, 'val_mae': val_maes})
		training_df.to_csv(f'training_curves_dep_GBM_{datetime.strftime(now_dt, "%Y_%m_%d_%H_%M_%S")}.csv', index=False)

	if return_best:
		return (best_mu, best_Sigma), train_hist_info
	else:
		return (detach(mu), detach(A @ A.T)), train_hist_info


def sample_dep_GBM(mu, Sigma, S_0, times, add_BM=True):
	"""
	Sample an independent Geometric Brownian Motion.

	Args:
		mu: GBM drift, shape (n_assets,).
		Sigma: the covariance matrix of Brownian Motions, shape (n_assets, n_assets)
		S_0: initial signal, shape (n_assets,).
		times: array of time steps since t_0.
		add_BM: whether to add Brownian Motion to sample, defaults to True.

	Returns: an array of [S_0 | sampled signals] with shape (S_0.shape[0], len(times) + 1).
	"""
	sigma_sq = np.linalg.norm(Sigma, axis=0, keepdims=False)
	avg_drift = mu - sigma_sq / 2
	if add_BM:
		bm = np.random.normal(size=(S_0.shape[0], times.shape[0]))

	S_ts = np.zeros((S_0.shape[0], len(times) + 1))
	S_ts[:, 0] = S_0
	for i in range(len(times)):
		dt = times[i] if i == 0 else times[i] - times[i - 1]
		vol = (dt ** 0.5 * Sigma @ bm[:, i - 1]) if add_BM else 0
		S_ts[:, i + 1] = S_ts[:, i] * np.exp(avg_drift * dt + vol)

	return S_ts


# Dependent Model w/ Covariates
def _get_weighted_MAE(dss, param_memos, theta, var_types, der_var_specs, covar_vars):
	_mae = 0.
	cum_weight = 0
	for ds, param_memo in zip(dss, param_memos):
		xarr = ds.to_dataarray()
		mu, Sigma = detach(param_memo[0]), detach(param_memo[1])
		ds_sim = sample_dep_cov_GBM(
			mu, Sigma, theta,
			xarr.sel(variable='signal').isel(time=0).to_numpy(), get_timestamps(xarr), ds, 0,
			var_types, der_var_specs, covar_vars,
			False
		)
		_mae += mae(xarr.sel(variable='signal').to_numpy().T[:, 1:], ds_sim[:, 1:]) * xarr.shape[0] * xarr.shape[1]
		cum_weight += xarr.shape[0] * xarr.shape[1]
	return _mae / cum_weight

def _Tensor_format_data(dX, dt, timestamps, Covs):
	return make_Tensor(dX), make_Tensor(dt), make_Tensor(timestamps), make_Tensor(Covs.transpose((2, 1, 0))[:, :-1, :])

def fit_dep_cov(
	train_ds, theta, covar_vars, mask_gen_fn, steps_per_batch,
	train_epochs=30,	l1_penalty=0., lr=1e-4, theta_lr=1e-5,
	mu=None, A=None, train_theta=True,
	**kwargs
) -> [torch.Tensor, torch.Tensor]:
	if mu is None or A is None:
		train_xarr = train_ds.to_dataarray()
		dX, dt, timestamps = get_dX_dt_timestamps(train_xarr.isel(time=slice(0, 20)))

		# get initial state from simple model
		mu_0, Sigma_0 = get_mle_dep_SRW_params(dX, dt)
		#   make Sigma_0 full rank a.s. by adding random noise. Note: was not working as the values added were too small
		# Sigma_0 = add_random_noise_to_Cov_mat(Sigma_0)
		#   alt. way to make sure Sigma_0 is full rank
		Sigma_0 += np.eye(Sigma_0.shape[0])*np.std(Sigma_0[np.triu_indices_from(Sigma_0, k=1)])
		A_0 = np.linalg.cholesky(Sigma_0).real

		mu, A = _init_mu_A_from_numpy(mu_0, A_0, mask_gen_fn)

	for i in range(train_epochs):
		lr_scale = 0.9**i
		if train_theta:
			optim = torch.optim.Adam([{'params': (mu, A), 'lr': lr*lr_scale}, {'params': (theta,)}], lr=theta_lr*lr_scale)
		else:
			optim = torch.optim.Adam((mu, A), lr=lr*lr_scale)

		for tr_batch in serve_xr_ds(train_ds, kwargs.get('batch_size', 20), kwargs.get('shuffle_mode', 'shuffle')):
			dX, dt, timestamps, Covs = _Tensor_format_data(*get_dX_dt_timestamps(tr_batch.to_dataarray()), tr_batch[covar_vars].to_dataarray().to_numpy().astype(np.float32))

			# train model params on this batch
			for _ in range(steps_per_batch):
				optim.zero_grad()
				Sigma = A @ A.T
				loss = get_dep_Cov_SRW_nll_loss(dX, dt, Covs, mu, Sigma, theta)
				loss += l1_penalty * torch.sum(torch.abs(theta))
				loss.backward()
				optim.step()
	return mu, A


def learn_dep_cov_MLE_theta(
	covar_vars, var_types, der_var_specs, train_dss: [xr.Dataset], val_dss: [xr.Dataset] = None,
	mask_gen_fn=triu_mask_gen_fn,
	n_epochs: int = 20,	lr=1e-3, theta_lr=1e-4, lr_decay=1.,
	steps_per_batch=20, val_freq=1,
	return_best=True, l1_penalty=0.,
	save_hist=False,
	**kwargs
) -> [np.ndarray, dict]:
	"""
	Calculate MLE MV-GBM model of a set of signals.

	Args:
		covar_vars ([str]): A required list of covariate variable names, each entry `covar_vars[i]` must be a xr.Variable in `market.xarray_ds`.
		var_types ({str; Metrics.VarType}): the types of variables in `covar_vars`.
	  der_var_specs ([DerivedVariables.DerivedVariable]): derived variable specifications to help impute Covariate data during evaluation.
	  train_dss ([xr.Dataset]): a list of training datasets, each dataset is specified by `var_types` and `der_var_specs`.
	  val_dss (optional, None or [xr.Dataset]): a list of validation datasets, each dataset is specified by `var_types` and `der_var_specs`.
		mask_gen_fn: `A` with shape `(market.get_num_assets(), market.get_num_assets())` -> binary tensor like `A`, A function that creates a mask to reduce the feature space of the covariance matrix.
		n_epochs: number of epochs of gradient ascent to run.
		lr (float): learning rate for local model parameters (`mu`, `Sigma`), defaults to `1e-3`.
		theta_lr (float): learning rate for global model parameters (`theta`, ), defaults to `1e-4`.
		lr_decay (float): learning rate decay factor, applied every 5 epochs, defaults to `1`.
		steps_per_batch: number of training steps/batch/epoch.
		val_freq: number of batches between validation metric calculations (excludes for the first 10 epochs when val metrics are always calculated).
		return_best: whether to return the best parameters (by MAE metric) or parameters from last iteration.
		l1_penalty: applied to the covariate coefficients.
		**kwargs:
		- any kwargs supported by `serve_xr_ds()`.

	Returns: mu_hat, Sigma_hat, theta_hat.
	"""

	train_info = [
		_Tensor_format_data(
			*get_dX_dt_timestamps(train_ds.to_dataarray()),
			train_ds[covar_vars].to_dataarray().to_numpy().astype(np.float32)
		) for train_ds in train_dss
	]
	if val_dss is not None:
		val_info = [
			_Tensor_format_data(
				*get_dX_dt_timestamps(val_ds.to_dataarray()),
				val_ds[covar_vars].to_dataarray().to_numpy().astype(np.float32)
			) for val_ds in val_dss
		]

	n_covars = len(covar_vars)

	train_losses = []
	train_maes = []
	val_losses = []
	val_maes = []
	epochs = []
	thetas = []

	theta_total = np.zeros((n_covars,), dtype=np.float32)
	weight_total = 0
	for (dX, dt, timestamps, Covs) in train_info:
		_, _, theta_i = get_mle_dep_Cov_SRW_params(detach(dX), detach(dt), detach(Covs))
		theta_total += theta_i
		weight_total += dX.shape[0]*dX.shape[1]
	theta_0 = (theta_total/weight_total).astype(np.float32)
	theta = make_param_Tensor(theta_0)

	best_theta = None
	best_metric = float('inf')

	train_param_memo = [None]*len(train_dss)

	for i in (p_bar := trange(n_epochs)):
		for ds_idx, train_ds in enumerate(train_dss):
			mu, A = train_param_memo[ds_idx] if train_param_memo[ds_idx] is not None else (None, None)

			mu, A = fit_dep_cov(
				train_ds, theta, covar_vars, mask_gen_fn, steps_per_batch,
				train_epochs=7,
				train_theta=True, lr=lr*lr_decay**(i//5), theta_lr=theta_lr*lr_decay**(i//5),
				l1_penalty=l1_penalty, mu=mu, A=A
			)
			train_param_memo[ds_idx] = mu, A

		# logging, evaluation metrics
		with torch.no_grad():
			# save per epoch eval metrics
			thetas.append(detach(theta))

			train_loss = sum(
				get_dep_Cov_SRW_nll_loss(
					dX, dt, Covs,
					mu, A@A.T, theta,
				) for (dX, dt, timestamps, Covs), (mu, A) in zip(train_info, train_param_memo)
			)
			train_losses.append(train_loss.item())
			if val_dss is not None:
				val_loss = sum(
					get_dep_Cov_SRW_nll_loss(
						dX, dt, Covs,
						mu, A@A.T, theta,
					) for (dX, dt, timestamps, Covs), (mu, A) in zip(val_info, train_param_memo))
				val_losses.append(val_loss.item())
			else:
				val_losses.append(np.nan)

			if (i < 10) or (i % val_freq == 0):
				epochs.append(i)
				train_maes.append(_get_weighted_MAE(train_dss, train_param_memo, thetas[-1], var_types, der_var_specs, covar_vars))
				val_maes.append(_get_weighted_MAE(val_dss, train_param_memo, thetas[-1], var_types, der_var_specs, covar_vars))

				p_bar.set_postfix({'train_loss': train_losses[-1], 'val_loss': val_losses[-1], 'train_MAE': train_maes[-1], 'val_MAE': val_maes[-1]})
				comp_metric = val_maes[-1] if val_dss is not None else train_maes[-1]
				if comp_metric < best_metric:
					best_metric = comp_metric
					best_theta = thetas[-1]
			else:
				epochs.append(np.nan)
				train_maes.append(np.nan)
				val_maes.append(np.nan)
				p_bar.set_postfix({'train_loss': train_losses[-1], 'val_loss': val_losses[-1]})

	train_hist_info = {
		'train_loss': train_losses,
		'val_loss': val_losses,
		'train_MAE': train_maes,
		'val_MAE': val_maes,
		'theta': thetas,
		'epochs': epochs
	}

	if save_hist:
		now_dt = datetime.now()
		with open(f'training_hist_dep_cov_GBM_{datetime.strftime(now_dt, "%Y_%m_%d_%H_%M_%S")}.pkl', 'wb') as file:
			pickle.dump(train_hist_info, file)
		training_df = pd.DataFrame({'epochs': epochs, 'train_loss': train_losses, 'val_loss': val_losses, 'train_mae': train_maes, 'val_mae': val_maes})
		training_df.to_csv(f'training_curves_dep_cov_GBM_{datetime.strftime(now_dt, "%Y_%m_%d_%H_%M_%S")}.csv', index=False)

	return best_theta if return_best else detach(theta), train_hist_info


def sample_dep_cov_GBM(
	mu, Sigma, theta,
	S_0, times, ds, t_start_idx,
	var_types, der_var_specs, covar_vars,
	add_BM=True
):
	"""
	Sample an independent Geometric Brownian Motion.

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

	Returns: a matrix [S_0 | sampled signals] with shape (S_0.shape[0], len(times) ).
	"""
	if len(ds['time']) == 1:
		return S_0.reshape(-1, 1).copy()
	t_start_idx = (t_start_idx + len(ds['time']) - 1) % (len(ds['time']) - 1)
	sigma_sq = np.linalg.norm(Sigma, axis=0, keepdims=False)**2
	avg_drift = mu - sigma_sq / 2
	if add_BM:
		bm = np.random.normal(size=(S_0.shape[0], times.shape[0]))

	S_ts = np.zeros((S_0.shape[0], len(times) + 1))
	S_ts[:, 0] = S_0
	for i in range(len(times)):
		dt = times[i] if i == 0 else times[i] - times[i - 1]
		vol = (dt ** 0.5 * Sigma @ bm[:, i - 1]) if add_BM else 0
		covars = ds[covar_vars].isel(time=t_start_idx + i).to_dataarray().astype(np.float32).to_numpy()
		covar_contr = np.einsum('Ca,C->a', covars, theta)
		S_ts[:, i + 1] = S_ts[:, i] + (avg_drift + covar_contr) * dt + vol
		if t_start_idx + i >= len(ds['time']):
			ds = append_xr_dataset(
				ds,
				S_ts[:, i + 1],
				dt,
				var_types,
				der_var_specs
			)

	return S_ts
