import numpy as np
import pandas as pd

from modeling_utils.py_utils import iqr_clip


def rmse(y_true, y_pred):
	return np.sqrt(np.mean(iqr_clip((y_pred - y_true)**2)))


def last_time_rmse(y_true, y_pred):
	return np.sqrt(np.mean(iqr_clip((y_pred[:, -1] - y_true[:, -1])**2)))


def mae(y_true, y_pred):
	return np.mean(iqr_clip(np.abs(y_pred - y_true) / y_true))


def last_time_mae(y_true, y_pred):
	return np.mean(iqr_clip(np.abs(y_pred[:, -1] - y_true[:, -1]) / y_true[:, -1]))


def _update_metrics(
	true_signal, sim_signal, split_idx,
	total_mae, total_rmse, total_last_mae, total_last_rmse, total_weight, total_last_weight
):
	"""
	Updates metrics based on true and simulated signals.

	Args:
		true_signal (np.array): shaped (n_stus, n_timestamps)
		sim_signal (np.array): shaped (n_sims, n_stus, n_timestamps)
		split_idx (one int from {0, 1, 2}): 0 for train, 1 for validation, 2 for test
		total_* ([float] or [int] length 3): total of some metric or metric weight, indexed by split_idx.

	Returns: None. Updates the `total_` parameters.
	"""
	n_sims, n_stus, n_times = sim_signal.shape
	sim_signal = sim_signal[:, :, 1:].transpose(1, 2, 0)
	total_mae[split_idx] += mae(true_signal[:, :, None], sim_signal) * n_stus * n_times
	total_rmse[split_idx] += rmse(true_signal[:, :, None], sim_signal) * n_stus * n_times
	total_last_mae[split_idx] += last_time_mae(true_signal[:, :, None], sim_signal) * n_stus
	total_last_rmse[split_idx] += last_time_rmse(true_signal[:, :, None], sim_signal) * n_stus
	total_weight[split_idx] += n_stus * n_times
	total_last_weight[split_idx] += n_stus


def evaluate_no_cov_model(model, markets, data_splits, **kwargs):
	"""
	Evaluate an Independent or Dependent (**No Covariate**) Model on a list of Markets.
	- Step 1: the model is fit on the train split for each market and train, validation metrics are calculated.
	- Step 2: the model is fit on the validation split for each market and test metrics are calculated.

	Args:
		model (models.Model): the model to evaluate.
		markets ([market.Markets]): each market in markets is a cohort's timeseries data.
		data_splits ([(xr.Dataset, xr.Dataset, xr.Dataset)]): the train, val, test splits, one per market in markets.
		**kwargs:
		- num_sims (int): defaults to 100.

	Returns: a dict of metrics {
		`split`_MAE: float, the mean absolute error (MAE) of all timestamps' signal predictions,
		`split`_last_MAE: float, the MAE at the last timestamp's signal predictions,
		`split`_RMSE: float, the root mean squared error (RMSE) of all timestamps' signal predictions,
		`split`_last_RMSE: float, the RMSE at the last timestamp's signal predictions
	} for each `split` in {'train', 'val', 'test'}.
	"""
	total_mae = [0., 0., 0.]
	total_rmse = [0., 0., 0.]
	total_last_mae = [0., 0., 0.]
	total_last_rmse = [0., 0., 0.]
	total_weight = [0, 0, 0]
	total_last_weight = [0, 0, 0]

	for i, market in enumerate(markets):
		mkt_i_train_ds, mkt_i_val_ds, mkt_i_test_ds = data_splits[i]
		# Step 1: the model is fit on the train split for each market and train, validation metrics are calculated.
		#   fit mu, Sigma for the market
		(mkt_mu, mkt_Sigma), mkt_train_hist = model.fit_local_params(
			market=None, train_ds=mkt_i_train_ds, val_ds=None,
			train_epochs=50, lr_decay=0.1**0.5, save_hist=False
		)
		if mkt_mu is None or mkt_Sigma is None:
			print('UwU')
		tr_dates_to_sim = tr_sim_dates = pd.Series(mkt_i_train_ds.time, name='dates')
		#   get S_0 for train subset
		tr_prev_signals = mkt_i_train_ds.to_dataarray().isel(time=0).sel(variable='signal').to_numpy()

		#   simulate train data
		sim_res = model.simulate(
			market, tr_prev_signals, tr_sim_dates, num_sims=kwargs.get('num_sims', 100),
			mu=mkt_mu, Sigma=mkt_Sigma
		)
		#   calculate train split metrics
		tr_true_signal = mkt_i_train_ds['signal'].isel(time=slice(1, None)).to_numpy().T
		_update_metrics(
			tr_true_signal, sim_res, 0,
			total_mae, total_rmse, total_last_mae, total_last_rmse, total_weight, total_last_weight
		)


		val_dates_to_sim = pd.Series(mkt_i_val_ds.time, name='dates')
		val_prev_time, val_prev_state = market.get_market_state_before_date(min(val_dates_to_sim))
		val_sim_dates = pd.concat((pd.Series(val_prev_time, name='dates'), val_dates_to_sim))
		#   get S_0 for val subset
		val_prev_signals = val_prev_state.sel(variable='signal').to_numpy()
		#   simulate the val subset
		sim_res = model.simulate(
			market, val_prev_signals, val_sim_dates, num_sims=kwargs.get('num_sims', 100),
			mu=mkt_mu, Sigma=mkt_Sigma
		)
		#   calculate val split metrics
		val_true_signal = mkt_i_val_ds['signal'].to_numpy().T
		_update_metrics(
			val_true_signal, sim_res, 1,
			total_mae, total_rmse, total_last_mae, total_last_rmse, total_weight, total_last_weight
		)


		# Step 2: the model is fit on the validation split for each market and test metrics are calculated.
		#   fit mu, Sigma for the market
		(val_mkt_mu, val_mkt_Sigma), mkt_train_hist = model.fit_local_params(
			market=None, train_ds=data_splits[i][1], val_ds=None,
			train_epochs=50, lr_decay=0.1**0.5, save_hist=False
		)

		#   find the times to simulate for the test subset
		test_dates_to_sim = pd.Series(mkt_i_test_ds.time, name='dates')
		test_prev_time, test_prev_state = market.get_market_state_before_date(min(test_dates_to_sim))
		test_sim_dates = pd.concat((pd.Series(test_prev_time, name='dates'), test_dates_to_sim))
		#   get S_0 for test subset
		test_prev_signals = test_prev_state.sel(variable='signal').to_numpy()
		#   simulate the test subset
		sim_res = model.simulate(
			market, test_prev_signals, test_sim_dates, num_sims=kwargs.get('num_sims', 100),
			mu=val_mkt_mu, Sigma=val_mkt_Sigma
		)
		#   calculate test split metrics
		test_true_signal = mkt_i_test_ds['signal'].to_numpy().T
		_update_metrics(
			test_true_signal, sim_res, 2,
			total_mae, total_rmse, total_last_mae, total_last_rmse, total_weight, total_last_weight
		)

	return {
		'train_MAE' : total_mae[0] / total_weight[0],
		'train_last_MAE' : total_last_mae[0] / total_last_weight[0],
		'train_RMSE' : total_rmse[0] / total_weight[0],
		'train_last_RMSE' : total_last_rmse[0] / total_last_weight[0],
		'val_MAE' : total_mae[1] / total_weight[1],
		'val_last_MAE' : total_last_mae[1] / total_last_weight[1],
		'val_RMSE' : total_rmse[1] / total_weight[1],
		'val_last_RMSE' : total_last_rmse[1] / total_last_weight[1],
		'test_MAE' : total_mae[2] / total_weight[2],
		'test_last_MAE' : total_last_mae[2] / total_last_weight[2],
		'test_RMSE' : total_rmse[2] / total_weight[2],
		'test_last_RMSE' : total_last_rmse[2] / total_last_weight[2],
	}

def evaluate_cov_model(model, theta, markets, data_splits, **kwargs):
	"""
	Evaluate an Independent or Dependent (**No Covariate**) Model on a list of Markets.
	- Step 1: the model is fit on the train split for each market and train, validation metrics are calculated.
	- Step 2: the model is fit on the validation split for each market and test metrics are calculated.

	Args:
		model (models.Model): the model to evaluate.
		theta (np.array, shape (n_covs, )): the learned covariate coefficients.
		markets ([market.Markets]): each market in markets is a cohort's timeseries data.
		data_splits ([(xr.Dataset, xr.Dataset, xr.Dataset)]): the train, val, test splits, one per market in markets.
		**kwargs:
		- num_sims (int): defaults to 100.

	Returns: a dict of metrics {
		`split`_MAE: float, the mean absolute error (MAE) of all timestamps' signal predictions,
		`split`_last_MAE: float, the MAE at the last timestamp's signal predictions,
		`split`_RMSE: float, the root mean squared error (RMSE) of all timestamps' signal predictions,
		`split`_last_RMSE: float, the RMSE at the last timestamp's signal predictions
	} for each `split` in {'train', 'val', 'test'}.
	"""
	total_mae = [0., 0., 0.]
	total_rmse = [0., 0., 0.]
	total_last_mae = [0., 0., 0.]
	total_last_rmse = [0., 0., 0.]
	total_weight = [0, 0, 0]
	total_last_weight = [0, 0, 0]

	for i, market in enumerate(markets):
		mkt_i_train_ds, mkt_i_val_ds, mkt_i_test_ds = data_splits[i]
		# Step 1: the model is fit on the train split for each market and train, validation metrics are calculated.
		#   fit mu, Sigma for the market
		mkt_mu, mkt_Sigma = model.fit_local_params(
			market=None, train_ds=mkt_i_train_ds, val_ds=None,
			train_epochs=30, steps_per_batch=20, theta=theta,
			covar_vars=model.covar_vars, lr_decay=0.1**0.5, save_hist=False
		)
		tr_dates_to_sim = tr_sim_dates = pd.Series(mkt_i_train_ds.time, name='dates')
		#   get S_0 for train subset
		tr_prev_signals = mkt_i_train_ds.to_dataarray().isel(time=0).sel(variable='signal').to_numpy()

		#   simulate train data
		sim_res = model.simulate(
			market, tr_prev_signals, tr_sim_dates, num_sims=kwargs.get('num_sims', 100),
			mu=mkt_mu, Sigma=mkt_Sigma, theta=theta
		)
		#   calculate train split metrics
		tr_true_signal = mkt_i_train_ds['signal'].isel(time=slice(1, None)).to_numpy().T
		_update_metrics(
			tr_true_signal, sim_res, 0,
			total_mae, total_rmse, total_last_mae, total_last_rmse, total_weight, total_last_weight
		)


		val_dates_to_sim = pd.Series(mkt_i_val_ds.time, name='dates')
		val_prev_time, val_prev_state = market.get_market_state_before_date(min(val_dates_to_sim))
		val_sim_dates = pd.concat((pd.Series(val_prev_time, name='dates'), val_dates_to_sim))
		#   get S_0 for val subset
		val_prev_signals = val_prev_state.sel(variable='signal').to_numpy()
		#   simulate the val subset
		sim_res = model.simulate(
			market, val_prev_signals, val_sim_dates, num_sims=kwargs.get('num_sims', 100),
			mu=mkt_mu, Sigma=mkt_Sigma, theta=theta
		)
		#   calculate val split metrics
		val_true_signal = mkt_i_val_ds['signal'].to_numpy().T
		_update_metrics(
			val_true_signal, sim_res, 1,
			total_mae, total_rmse, total_last_mae, total_last_rmse, total_weight, total_last_weight
		)


		# Step 2: the model is fit on the validation split for each market and test metrics are calculated.
		#   fit mu, Sigma for the market
		val_mkt_mu, val_mkt_Sigma = model.fit_local_params(
			market=None, train_ds=data_splits[i][1], val_ds=None,
			train_epochs=30, steps_per_batch=20, theta=theta,
			covar_vars=model.covar_vars, lr_decay=0.1**0.5, save_hist=False
		)

		#   find the times to simulate for the test subset
		test_dates_to_sim = pd.Series(mkt_i_test_ds.time, name='dates')
		test_prev_time, test_prev_state = market.get_market_state_before_date(min(test_dates_to_sim))
		test_sim_dates = pd.concat((pd.Series(test_prev_time, name='dates'), test_dates_to_sim))
		#   get S_0 for test subset
		test_prev_signals = test_prev_state.sel(variable='signal').to_numpy()
		#   simulate the test subset
		sim_res = model.simulate(
			market, test_prev_signals, test_sim_dates, num_sims=kwargs.get('num_sims', 100),
			mu=val_mkt_mu, Sigma=val_mkt_Sigma, theta=theta
		)
		#   calculate test split metrics
		test_true_signal = mkt_i_test_ds['signal'].to_numpy().T
		_update_metrics(
			test_true_signal, sim_res, 2,
			total_mae, total_rmse, total_last_mae, total_last_rmse, total_weight, total_last_weight
		)

	return {
		'train_MAE' : total_mae[0] / total_weight[0],
		'train_last_MAE' : total_last_mae[0] / total_last_weight[0],
		'train_RMSE' : total_rmse[0] / total_weight[0],
		'train_last_RMSE' : total_last_rmse[0] / total_last_weight[0],
		'val_MAE' : total_mae[1] / total_weight[1],
		'val_last_MAE' : total_last_mae[1] / total_last_weight[1],
		'val_RMSE' : total_rmse[1] / total_weight[1],
		'val_last_RMSE' : total_last_rmse[1] / total_last_weight[1],
		'test_MAE' : total_mae[2] / total_weight[2],
		'test_last_MAE' : total_last_mae[2] / total_last_weight[2],
		'test_RMSE' : total_rmse[2] / total_weight[2],
		'test_last_RMSE' : total_last_rmse[2] / total_last_weight[2],
	}
