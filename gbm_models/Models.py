from abc import ABC, abstractmethod

import numpy as np

from gbm_models.utils import *
from markets.Markets import XarrayMarket
from py_utils.py_utils import last_index_lt_1D


class Model(ABC):
  def __init__(self, market):
    self.market = market
    self.N_ASSETS = len(self.market.assets)

  @abstractmethod
  def estimate_parameters(self, **kwargs):
    pass

  @abstractmethod
  def simulate(self, dates, num_sims, **kwargs):
    """
    simulate future market behavior

    Args:
      dates: a timeseries of dates
      num_sims: number of simulations
      kwargs: optional keyword arguments
        add_BM: bool indicating whether to add Brownian Motion (randomness) to simulation

    Returns: a 3D numpy array [num_sims x |students| x |dates|]
    """
    pass


class IndependentModel(Model):
  def __init__(self, market):
    super().__init__(market)

    self.drifts = np.zeros((self.N_ASSETS,), dtype=np.float32)
    self.vols = np.zeros((self.N_ASSETS,), dtype=np.float32)

  def estimate_parameters(self, **kwargs):
    for i, asset in enumerate(self.market.assets):
      self.drifts[i], self.vols[i] = get_indep_MLE_params(asset.ts_df)

  def simulate(self, dates, num_sims, **kwargs):
    sim_res = np.zeros((num_sims, self.N_ASSETS, len(dates)+1))

    for n_sim in range(num_sims):
      for i, asset in enumerate(self.market.assets):
        ts = (dates - asset.ts_df.time.max()).dt.days

        sim_res[n_sim, i, :] = sample_indep_GBM(
          self.drifts[i],
          self.vols[i],
          asset.ts_df.signal.iloc[-1],
          ts.values,
          kwargs.get('add_BM', True)
        )
    return sim_res


class DependentModel(Model):
  def __init__(self, market):
    if not isinstance(market, XarrayMarket):
      raise TypeError(f'DependentModel requires market to be a XarrayMarket, got {type(market)}')
    super().__init__(market)

    self.drifts = np.zeros((self.N_ASSETS,), dtype=np.float32)
    self.Sigma = np.zeros((self.N_ASSETS, self.N_ASSETS), dtype=np.float32)

  def estimate_parameters(self, **kwargs):
    self.drifts, self.Sigma = get_dep_MLE_params(self.market, **kwargs)
    self.vols = np.linalg.norm(self.Sigma, axis=0, keepdims=True)

  def simulate(self, dates, num_sims, **kwargs):
    """
    simulate future market behavior

    Args:
      dates: datetime array
      num_sims: number of simulations to run
      **kwargs: optional arguments
        add_BM: bool indicating whether to add Brownian Motion (randomness) to simulation

    Returns: simulation results in 3D np array with shape (num_sims, self.N_ASSETS, len(dates) + 1)

    """
    sim_res = np.zeros((num_sims, self.N_ASSETS, len(dates) + 1))
    market_times = self.market.xarray.time.values
    market_t_index = last_index_lt_1D(market_times, min(dates))
    # calculate # of days since the last recorded date before new dates
    ts = (dates - market_times[market_t_index]).dt.days

    for n_sim in range(num_sims):
      sim_res[n_sim, :, :] = sample_corr_GBM(
        self.drifts, self.Sigma,
        self.market.xarray.sel(time=market_times[market_t_index], variable='signal').to_numpy(),
        ts,
        add_BM=kwargs.get('add_BM', True)
      )

    return sim_res

  def get_correlation_mat(self):
    return self.Sigma


class CovariateModel(Model, ABC):
  @abstractmethod
  def summarize_covariate_distributions(self):
    pass


class IndependentCovariateModel(CovariateModel, IndependentModel):
  def estimate_parameters(self):
    raise NotImplementedError("indep covar model")

  def simulate(self, dates, num_sims, **kwargs):
    raise NotImplementedError("indep covar model")

  def summarize_covariate_distributions(self):
    raise NotImplementedError("indep covar model")


class DependentCovariateModel(CovariateModel, DependentModel):
  def estimate_parameters(self):
    raise NotImplementedError("corr covar model")

  def simulate(self, dates, num_sims, **kwargs):
    raise NotImplementedError("corr covar model")

  def summarize_covariate_distributions(self):
    raise NotImplementedError("corr covar model")
