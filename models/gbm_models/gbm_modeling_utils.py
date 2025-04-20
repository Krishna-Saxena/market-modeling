from datetime import datetime
import pickle

import numpy as np
import pandas as pd
import xarray as xr
import torch
from tqdm import trange
from matplotlib import pyplot as plt

from metrics.EvalMetrics import mae
from modeling_utils.torch_utils import get_zero_grad_hook
from modeling_utils.xr_utils import append_xr_dataset, get_covar_indices_and_names

SEED = 802073711

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
    `mu`: drift of GBM.
    `sigma_sq`: variance of GBM.
    `S_0`: initial signal.
    `times`: array of time steps since t_0.
    `add_BM`: whether to add Brownian Motion to sample.

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
  avg_drift = mu_hat - sigma_sq_hat / 2

  dt = (ts_df['time'] - ts_df['time'].shift(1)).dt.days.values
  sim_results = np.tile(ts_df[col_to_fill].values.reshape(-1,1), (1, num_sims))
  for i in range(1, ts_df.shape[0]):
    if np.isnan(ts_df[col_to_fill].iloc[i]):
      sim_results[i, :] = sim_results[i-1, :]*np.exp(dt[i]*avg_drift + (dt[i]*sigma_sq_hat)**0.5*np.random.randn(num_sims))
  ts_df[col_to_fill] = np.mean(sim_results, axis=1)
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
  avg_drift = mu_hat - sigma_sq_hat / 2

  dt = (ts_df['time'].shift(-1) - ts_df['time']).dt.days.values
  sim_results = np.tile(ts_df[col_to_fill].values.reshape(-1,1), (1, num_sims))
  for i in range(ts_df.shape[0]-1, 0, -1):
    if np.isnan(ts_df[col_to_fill].iloc[i-1]):
      sim_results[i-1, :] = sim_results[i, :]*np.exp(-dt[i-1]*avg_drift - (dt[i-1]*sigma_sq_hat)**0.5*np.random.randn(num_sims))
  ts_df[col_to_fill] = np.mean(sim_results, axis=1)
  return ts_df

# Dependent Model
def _get_init_dep_MLE_params(dX, dt):
  delta_X = np.sum(dX, axis=1)
  delta_t = np.sum(dt)

  mu_tilde = delta_X / delta_t
  scaled_mu = np.einsum('a,t->at', mu_tilde, dt)
  sigma_tilde = 1 / len(dt) * np.einsum('ij,kj->ik', (dX-scaled_mu)/dt[None, :], dX - scaled_mu)
  return mu_tilde, sigma_tilde

def _get_nll_loss_dep(dX, dt, mu, Sigma):
  drift = mu - torch.linalg.norm(Sigma, dim=0)/2
  scaled_drift = torch.einsum('a,t->at', drift, dt)

  z = torch.linalg.inv(Sigma)@(dX - scaled_drift)/torch.sqrt(dt[None, :])

  return torch.einsum('at,at->', z, z)/2

def _add_random_noise_to_Cov_mat(Sigma, adjustment=100):
  # 1. set mu = [-1, 1]
  mu = np.array([-1, 1])
  # 2. run one iter of K-means clustering on triu(Sigma) with seed mu
  cov_vals = Sigma[np.triu_indices_from(Sigma, k=1)]
  assignments = np.argmin((cov_vals[:, None]-mu[None, :])**2, axis=1)
  # 3. get vars of each mode
  var0 = 0. if np.allclose(assignments, 1) else np.var(cov_vals[assignments == 0])
  var1 = 0. if np.allclose(assignments, 0) else np.var(cov_vals[assignments == 1])
  # 4. let sig2_hat = weighted avg of the modes' vars, weighted by members of each mode
  sig2_hat = (var0*np.sum(assignments == 0) + var1*np.sum(assignments == 1))/(assignments.shape[0])
  # 5. add N(0, sig2_hat*I) noise to Sigma
  rng = np.random.RandomState(SEED)
  return Sigma + rng.normal(0, sig2_hat**0.5/adjustment, Sigma.shape)

def get_dep_MLE_params(market, n_epochs=200, return_best=True):
  """
  Calculate MLE MV-GBM model of a set of signals.

  Args:
    `market`: an XarrayMarket object.
    `n_epochs`: number of epochs of gradient ascent to run.
    `return_best`: whether to return the best parameters (by MAE metric) or parameters from last iteration.

  Returns: mu_hat, Sigma_hat.
  """
  # dt has shape (n_ts, ) with 1st entry = NaN
  xarr = market.get_dataarray()
  dt = (xarr.time - xarr.time.shift({'time':1})).dt.days.values
  X = np.log(xarr.sel(variable='signal').astype(np.float32))
  # dX has shape (n_assets, n_ts) with 1st col = NaN
  dX = (X - X.shift({'time': 1})).to_numpy().T

  # remove NaN obs
  dt = dt[1:]
  dX = dX[:, 1:]

  # get initial state from simple model
  mu_0, Sigma_0 = _get_init_dep_MLE_params(dX, dt)
  Sigma_0 = _add_random_noise_to_Cov_mat(Sigma_0)
  A_0 = np.linalg.cholesky(Sigma_0).real

  mu = torch.tensor(mu_0, requires_grad=True)
  A = torch.tensor(A_0, requires_grad=True)
  # keep A as a lower triangular matrix to reduce feature space
  #   https://discuss.pytorch.org/t/creating-a-triangular-weight-matrix-with-a-hidden-layer-instead-of-a-fully-connected-one/71427/
  mask = torch.triu(torch.ones_like(A))
  A.register_hook(get_zero_grad_hook(mask))

  dX = torch.tensor(dX)
  dt = torch.tensor(dt)

  # run optimization loop
  optim = torch.optim.Adam((mu, A), lr=1e-3)
  #   reduce the LR 10% every 100 epochs
  torch.optim.lr_scheduler.StepLR(optim, step_size=50, gamma=0.1**0.5)
  losses = [0] * n_epochs
  maes = [0] * n_epochs

  best_mu, best_Sigma = None, None
  best_metric = float('inf')

  for i in range(n_epochs):
    optim.zero_grad()
    Sigma = A@A.T
    loss = _get_nll_loss_dep(dX, dt, mu, Sigma)
    loss.backward()
    optim.step()

    # save per epoch eval metrics
    losses[i] = loss.item()
    sim = sample_dep_GBM(
      mu.detach().numpy(), Sigma.detach().numpy(),
      xarr.sel(variable='signal').isel(time=0).to_numpy(),
      torch.cumsum(dt, 0).numpy()
    )
    # note [:, 1:] because the 1st col of sample_corr_GBM's result is S_0
    maes[i] = mae(xarr.sel(variable='signal').to_numpy().T[:, 1:], sim[:, 1:])

    if maes[i] < best_metric:
      best_metric = maes[i]
      best_mu = mu.detach().numpy()
      best_Sigma = Sigma.detach().numpy()

  plt.subplot(1, 2, 1)
  plt.plot(losses, '-o')
  plt.title('loss curve')

  plt.subplot(1, 2, 2)
  plt.plot(maes, '-o')
  plt.title('MAE metric')
  plt.tight_layout()

  if return_best:
    return best_mu, best_Sigma
  else:
    return mu.detach().numpy(), (A@A.T).detach().numpy()

def sample_dep_GBM(mu, Sigma, S_0, times, add_BM=True):
  """
  Sample an independent Geometric Brownian Motion.

  Args:
    `mu`: GBM drift, shape (n_assets,).
    `Sigma`: the covariance matrix of Brownian Motions, shape (n_assets, n_assets)
    `S_0`: initial signal, shape (n_assets,).
    `times`: array of time steps since t_0.
    `add_BM`: whether to add Brownian Motion to sample.

  Returns: an array of [S_0 | sampled signals] with shape (S_0.shape[0], len(times) + 1).
  """
  sigma_sq = np.linalg.norm(Sigma, axis=0, keepdims=False)
  avg_drift = mu - sigma_sq / 2
  bm = np.random.normal(size=(S_0.shape[0], times.shape[0]))

  S_ts = np.zeros((S_0.shape[0], len(times) + 1))
  S_ts[:, 0] = S_0
  for i in range(len(times)):
    dt = times[i] if i == 0 else times[i] - times[i - 1]
    vol = (dt ** 0.5 * Sigma@bm[:, i - 1]) if add_BM else 0
    S_ts[:, i + 1] = S_ts[:, i] * np.exp(avg_drift * dt + vol)

  return S_ts

# Dependent Model w/ Covariates
def _get_nll_loss_dep_cov(dX, dt, mu, Sigma, theta, Covs):
  drift = mu - torch.linalg.norm(Sigma, dim=0)/2
  scaled_drift = torch.einsum('a,t->at', drift, dt) + torch.einsum('C,atC,t->at', theta, Covs, dt)

  z = torch.linalg.inv(Sigma)@(dX - scaled_drift)/torch.sqrt(dt[None, :])

  return torch.einsum('at,at->', z, z)/2

def get_dep_cov_MLE_params(market, covar_vars, n_epochs=200, return_best=True, l1_penalty=0.):
  """
  Calculate MLE MV-GBM model of a set of signals.

  Args:
    `market`: an XarrayMarket object.
    `covar_vars` [str]: A required list of covariate variable names, each entry `covar_vars[i]` must be a xr.Variable in `market.xarray_ds`.
    `n_epochs`: number of epochs of gradient ascent to run.
    `return_best`: whether to return the best parameters (by MAE metric) or parameters from last iteration.

  Returns: mu_hat, Sigma_hat, theta_hat.
  """
  n_covars = len(covar_vars)

  # dt has shape (n_ts, ) with 1st entry = NaN
  xarr = market.get_dataarray()
  dt = (xarr.time - xarr.time.shift({'time':1})).dt.days.values.astype(np.float32)
  X = np.log(xarr.sel(variable='signal').astype(np.float32))
  # dX has shape (n_assets, n_ts) with 1st col = NaN
  dX = (X - X.shift({'time': 1})).to_numpy().T

  covar_indices, _ = get_covar_indices_and_names(market.xarray_ds, covar_vars)
  # Covs has shape (n_assets, n_ts, n_covars)
  Covs = xarr.isel(variable=covar_indices).astype(np.float32).to_numpy()
  Covs = Covs.transpose((2, 1, 0))

  # shift time axes to remove NaN obs
  dt = dt[1:]
  dX = dX[:, 1:]
  #   shift Covs differently to avoid EOD information from influencing day's prediction
  Covs = Covs[:, :-1, :]

  times = np.cumsum(dt)

  # get initial state from simple model
  mu_0, Sigma_0 = _get_init_dep_MLE_params(dX, dt)
  Sigma_0 = _add_random_noise_to_Cov_mat(Sigma_0)
  A_0 = np.linalg.cholesky(Sigma_0).real.astype(np.float32)
  # (n_assets, n_timesteps, n_covars) = Covs.shape
  # theta_0, mu_0 = np.linalg.lstsq(Covs.reshape(n_assets*n_timesteps, n_covars), mu_0)
  theta_0 = (np.random.randn(n_covars)/n_covars**2).astype(mu_0.dtype)

  mu = torch.tensor(mu_0, requires_grad=True)
  A = torch.tensor(A_0, requires_grad=True)
  # keep A as a lower triangular matrix to reduce feature space
  #   https://discuss.pytorch.org/t/creating-a-triangular-weight-matrix-with-a-hidden-layer-instead-of-a-fully-connected-one/71427/
  mask = torch.triu(torch.ones_like(A))
  A.register_hook(get_zero_grad_hook(mask))
  theta = torch.tensor(theta_0, requires_grad=True)

  dX = torch.tensor(dX)
  dt = torch.tensor(dt)
  Covs_pt = torch.tensor(Covs)

  # run optimization loop
  optim = torch.optim.Adam((mu, A, theta), lr=1e-3)
  #   reduce the LR 10% every 100 epochs
  torch.optim.lr_scheduler.StepLR(optim, step_size=50, gamma=0.1**0.5)
  i_s = []
  losses = []
  maes = []
  opt_hist = {}

  best_mu, best_Sigma, best_theta = None, None, None
  best_metric = float('inf')

  for i in (pbar := trange(n_epochs)):
    optim.zero_grad()
    Sigma = A@A.T
    loss = _get_nll_loss_dep_cov(dX, dt, mu, Sigma, theta, Covs_pt)
    loss += l1_penalty*torch.sum(torch.abs(theta))
    loss.backward()
    optim.step()

    if (i < 10) or (i % 3 == 0 and i < 50) or (i % 10 == 0):
      i_s.append(i)
      # save per epoch eval metrics
      losses.append(loss.item())
      sim = sample_dep_cov_GBM(
        mu.detach().numpy(), Sigma.detach().numpy(), theta.detach().numpy(),
        xarr.sel(variable='signal').isel(time=0).to_numpy(),
        times,
        market.xarray_ds, 0,
        list(market.derived_variables.keys()), covar_indices
      )
      # note [:, 1:] because the 1st col of sample_corr_GBM's result is S_0
      maes.append(mae(xarr.sel(variable='signal').to_numpy().T[:, 1:], sim[:, 1:]))

      pbar.set_postfix({'loss': losses[-1], 'MAE': maes[-1]})

      opt_hist[i] = (losses[-1], maes[-1], mu.detach().numpy(), Sigma.detach().numpy(), theta.detach().numpy())

      if maes[-1] < best_metric:
        best_metric = maes[-1]
        best_mu = mu.detach().numpy()
        best_Sigma = Sigma.detach().numpy()
        best_theta = theta.detach().numpy()

  with open(f'training_hist_{datetime.strftime(datetime.now(), "%Y_%m_%d_%H_%M_%S")}.pkl', 'wb') as file:
    pickle.dump(opt_hist, file)
  training_df = pd.DataFrame({'epochs':i_s, 'loss': losses, 'mae': maes})
  training_df.to_csv(f'training_data_{datetime.strftime(datetime.now(), "%Y_%m_%d_%H_%M_%S")}.csv', index=False)

  if return_best:
    return best_mu, best_Sigma, best_theta
  else:
    return mu.detach().numpy(), (A@A.T).detach().numpy(), theta.detach().numpy()

def sample_dep_cov_GBM(mu, Sigma, theta, S_0, times, ds, t_start_idx, der_var_specs, covar_indices, add_BM=True):
  """
  Sample an independent Geometric Brownian Motion.

  Args:
    `mu`: GBM drift, shape (n_assets,).
    `Sigma`: the covariance matrix of Brownian Motions, shape (n_assets, n_assets).
    `theta`: weights for covariates, shape (n_covars,).
    `S_0`: initial signal, shape (n_assets,).
    `times`: array of time steps since t_0.
    `ds`: xarray.Dataset of full market history, shape (n_assets, n_ts, n_covars).
    `t_start_idx`: index of `ds.sel(variable=COVAR_VARS)` corresponding to covars at t_0, will be mapped onto [0, len(ds['time'])-1). Generally, set t_start_idx to 0 to test GoF and t_start_idx to -1 to forecast.
    `der_var_specs`: a list of DerivedVariable specifications.
    `covar_indices`: the locations of covariate variables along `ds`'s variable axis, shape/len (n_covars,).
    `add_BM`: whether to add Brownian Motion when generating samples.

  Returns: an array of [S_0 | sampled signals] with shape (S_0.shape[0], len(times) + 1).
  """
  t_start_idx = (t_start_idx + len(ds['time'])-1) % (len(ds['time'])-1)
  sigma_sq = np.linalg.norm(Sigma, axis=0, keepdims=False)
  avg_drift = mu - sigma_sq / 2
  bm = np.random.normal(size=(S_0.shape[0], times.shape[0]))

  S_ts = np.zeros((S_0.shape[0], len(times) + 1))
  S_ts[:, 0] = S_0
  for i in range(len(times)):
    dt = times[i] if i == 0 else times[i] - times[i - 1]
    vol = (dt ** 0.5 * Sigma@bm[:, i - 1]) if add_BM else 0
    covars = ds.isel(time=t_start_idx+i).to_dataarray().isel(variable=covar_indices).astype(np.float32).to_numpy()
    covar_contr = np.einsum('Ca,C->a', covars, theta)
    S_ts[:, i + 1] = S_ts[:, i] * np.exp((avg_drift + covar_contr) * dt + vol)
    if t_start_idx+i >= len(ds['time']):
      ds = append_xr_dataset(
        ds,
        S_ts[:, i + 1],
        dt,
        der_var_specs
      )

  return S_ts