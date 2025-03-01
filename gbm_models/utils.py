import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt

from markets.Markets import Market, XarrayMarket
from metrics.EvalMetrics import mae


def get_indep_MLE_params(ts_df: pd.DataFrame):
  """
  Calculates the MLE GBM parameters using a single timeseries

  Args:
    ts_df: a pd.DataFrame with columns (at least) `time` and `signal`

  Returns: mu_hat, sigma_sq_hat, the MLE drift and volatility

  """
  t = ts_df.time - ts_df['time'].min()
  dt = (t - t.shift(1)).dt.days.values
  signal = ts_df['signal'].values

  n = signal.shape[0]
  log_prices = np.log(signal)
  delta_X = log_prices[-1] - log_prices[0]
  delta_t = np.nansum(dt)

  sigma_sq_hat = -1 / n * delta_X ** 2 / delta_t + np.nanmean((log_prices[1:] - log_prices[:-1]) ** 2 / dt[1:])
  mu_hat = delta_X / delta_t + 0.5 * sigma_sq_hat

  return mu_hat, sigma_sq_hat


def sample_indep_GBM(mu, sigma_sq, S_0, ts, add_BM=True):
  """
  Sample an independent Geometric Brownian Motion

  Args:
    mu: drift of GBM
    sigma_sq: variance of GBM
    S_0: initial signal
    ts: array of time steps since t_0
    add_BM: whether to add Brownian Motion to sample

  Returns: an array of sampled signals

  """
  avg_drift = mu - sigma_sq / 2
  vols = np.random.normal(size=ts.shape)

  S_ts = np.zeros((len(ts) + 1,))
  S_ts[0] = S_0
  for i in range(len(ts)):
    dt = ts[i] if i == 0 else ts[i] - ts[i - 1]
    vol = (sigma_sq * dt) ** 0.5 * vols[i - 1] if add_BM else 0
    S_ts[i + 1] = S_ts[i] * np.exp(avg_drift * dt + vol)

  return S_ts


def _get_init_dep_MLE_params(dX, dt):
  delta_X = np.sum(dX, axis=1)
  delta_t = np.sum(dt)

  mu_tilde = delta_X / delta_t
  scaled_mu = np.einsum('a,t->at', mu_tilde, dt)
  sigma_tilde = 1 / len(dt) * np.einsum('ij,kj->ik', (dX-scaled_mu)/dt[None, :], dX - scaled_mu)
  return mu_tilde, sigma_tilde


def _get_nll_loss_dep_no_covar(dX, dt, mu, Sigma):
  offset = mu - torch.linalg.norm(Sigma, dim=0)/2
  scaled_offset = torch.einsum('a,t->at', offset, dt)

  z = torch.linalg.inv(Sigma)@(dX - scaled_offset)/torch.sqrt(dt[None, :])

  return torch.einsum('at,at->', z, z)/2

def _get_zero_grad_hook(mask):
  def hook(grad):
    return grad * mask
  return hook


def get_dep_MLE_params(market: Market, n_iters=200, return_best=True):
  """
  Calculate MLE MV-GBM model of a set of signals

  Args:
    market: an XarrayMarket object

  Returns: mu_hat, A_hat

  """
  if not isinstance(market, XarrayMarket):
    raise TypeError(f'Dependent/Covar/Heir Models only support XarrayMarket, got {type(market)}')

  # dt has shape (n_ts, ) with 1st entry = NaN
  dt = (market.xarray.time - market.xarray.time.shift({'time':1})).dt.days.values
  X = np.log(market.xarray)
  # dX has shape (n_assets, n_ts) with 1st col = NaN
  dX = (X - X.shift({'time': 1})).sel(variable='signal').to_numpy().T

  # remove NaN obs
  dt = dt[1:]
  dX = dX[:, 1:]

  # get initial state from simple model
  mu_0, Sigma_0 = _get_init_dep_MLE_params(dX, dt)
  A_0 = np.linalg.cholesky(Sigma_0).real

  mu = torch.tensor(mu_0, requires_grad=True)
  A = torch.tensor(A_0, requires_grad=True)
  # keep A as a lower triangular matrix to reduce feature space
  #   https://discuss.pytorch.org/t/creating-a-triangular-weight-matrix-with-a-hidden-layer-instead-of-a-fully-connected-one/71427/
  mask = torch.triu(torch.ones_like(A))
  A.register_hook(_get_zero_grad_hook(mask))

  dX = torch.tensor(dX)
  dt = torch.tensor(dt)

  # run optimization loop
  optim = torch.optim.Adam((mu, A), lr=1e-3)
  #   reduce the LR 10% every 100 epochs
  torch.optim.lr_scheduler.StepLR(optim, step_size=50, gamma=0.1**0.5)
  losses = [0] * n_iters
  maes = [0] * n_iters

  best_mu, best_Sigma = None, None
  best_metric = float('inf')

  for i in range(n_iters):
    optim.zero_grad()
    Sigma = A@A.T
    loss = _get_nll_loss_dep_no_covar(dX, dt, mu, Sigma)
    loss.backward()
    optim.step()

    # save per epoch eval metrics
    losses[i] = loss.item()
    sim = sample_corr_GBM(
      mu.detach().numpy(), Sigma.detach().numpy(),
      market.xarray.sel(variable='signal').isel(time=0).to_numpy(),
      np.cumsum(dt).numpy()
    )
    # note [:, 1:] because the 1st col of sample_corr_GBM's result is S_0
    maes[i] = mae(market.xarray.sel(variable='signal').to_numpy().T[:, 1:], sim[:, 1:])

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
    return mu.detach().numpy(), Sigma.detach().numpy()

def sample_corr_GBM(mu, Sigma, S_0, ts, add_BM=True):
  """
  Sample an independent Geometric Brownian Motion

  Args:
    mu: drift of GBM
    A: sqrt of covariance matrix
    S_0: initial signal
    ts: array of time steps since t_0
    add_BM: whether to add Brownian Motion to sample

  Returns: an array of [S_0 | sampled signals] with shape (S_0.shape[0], len(ts) + 1)

  """
  sigma_sq = np.linalg.norm(Sigma, axis=0, keepdims=False)
  avg_drift = mu - sigma_sq / 2
  bm = np.random.normal(size=(S_0.shape[0], ts.shape[0]))

  S_ts = np.zeros((S_0.shape[0], len(ts) + 1))
  S_ts[:, 0] = S_0
  for i in range(len(ts)):
    dt = ts[i] if i == 0 else ts[i] - ts[i - 1]
    vol = (dt ** 0.5 * Sigma@bm[:, i - 1]) if add_BM else 0
    S_ts[:, i + 1] = S_ts[:, i] * np.exp(avg_drift * dt + vol)

  return S_ts