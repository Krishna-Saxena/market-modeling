import numpy as np


def get_indep_MLE_params(timeseries):
  """
  Calculate MLE GBM model of a single signal

  Args:
    timeseries: a metrics.TimeSeries object

  Returns: mu_hat, sigma_sq_hat

  """
  t = timeseries.time - timeseries.time.min()
  dt = (t - t.shift(1)).dt.days.values
  signal = timeseries.signal.values

  n = signal.shape[0]
  log_prices = np.log(signal)
  delta_X = log_prices[-1] - log_prices[0]
  delta_t = np.nansum(dt)

  sigma_sq_hat = -1 / n * delta_X ** 2 / delta_t + \
                 np.nanmean((log_prices[1:] - log_prices[:-1]) ** 2 / dt[1:])
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

  S_ts = np.zeros((len(ts)+1, ))
  S_ts[0] = S_0
  for i in range(len(ts)):
    dt = ts[i] if i == 0 else ts[i] - ts[i-1]
    vol = (sigma_sq * dt)**0.5*vols[i-1] if add_BM else 0
    S_ts[i+1] = S_ts[i]*np.exp(avg_drift*dt + vol)

  # S_ts = np.concatenate(
  #   ([S_0],
  #    S_0 * np.exp(avg_drift * ts + np.random.normal(loc=0, scale=sigma_sq * ts)))
  # )

  return S_ts


def get_corr_MLE_params(timeseries):
  """
  Calculate MLE MV-GBM model of a set of signals

  Args:
    timeeries: a list of metrics.Timeseries objects, 1 per student

  Returns: mu_hat, sigma_sq_hat, Cov_hat

  """
  raise NotImplementedError('get_corr_MLE_params')

def sample_corr_GBM(mu, sigma_sq, Cov, S_0, ts, add_BM=True):
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
  raise NotImplementedError('sample_indep_GBM')
