import numpy as np


def get_indep_MLE_params(timeseries):
  """
  Calculate MLE GBM model of a single signal

  Args:
    timeseries: a metrics.TimeSeries object

  Returns: mu_hat, sigma_sq_hat

  """
  raise NotImplementedError('get_indep_MLE_params')


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
  raise NotImplementedError('sample_indep_GBM')


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
