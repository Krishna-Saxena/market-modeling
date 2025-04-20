from abc import ABC, abstractmethod

from models.Models import Model
from markets.Markets import Market
from models.gbm_models.gbm_modeling_utils import *

class IndependentModel(Model):
  def __init__(self, market: Market):
    super().__init__(market)

    self.drifts = np.zeros((self.N_ASSETS,), dtype=np.float32)
    self.vols = np.zeros((self.N_ASSETS,), dtype=np.float32)

  def estimate_parameters(self, **kwargs):
    for i in range(self.market.get_num_assets()):
      self.drifts[i], self.vols[i] = get_indep_MLE_params(self.market.xarray_ds.isel(ID=i))

  def simulate(self, S_0, dates, num_sims, **kwargs):
    sim_res = np.zeros((num_sims, self.N_ASSETS, len(dates)))
    ts = (dates - dates.iloc[0]).dt.days

    for n_sim in range(num_sims):
      for i in range(self.market.get_num_assets()):
        sim_res[n_sim, i, :] = sample_indep_GBM(
          self.drifts[i],
          self.vols[i],
          S_0[i],
          ts.values[1:],
          kwargs.get('add_BM', True)
        )
    return sim_res


class DependentModel(Model):
  def __init__(self, market: Market, **kwargs):
    super().__init__(market, **kwargs)

    self.drifts = np.zeros((self.N_ASSETS,), dtype=np.float32)
    self.Sigma = np.zeros((self.N_ASSETS, self.N_ASSETS), dtype=np.float32)

  def estimate_parameters(self, **kwargs):
    self.drifts, self.Sigma = get_dep_MLE_params(self.market, **kwargs)
    self.vols = np.linalg.norm(self.Sigma, axis=0, keepdims=True)

  def simulate(self, S_0, dates, num_sims, **kwargs):
    sim_res = np.zeros((num_sims, self.N_ASSETS, len(dates)))
    ts = (dates - dates.iloc[0]).dt.days

    for n_sim in range(num_sims):
      sim_res[n_sim, :, :] = sample_dep_GBM(
        self.drifts, self.Sigma,
        S_0,
        ts[1:],
        add_BM=kwargs.get('add_BM', True)
      )

    return sim_res

  def get_correlation_mat(self):
    return self.Sigma


class CovariateModel(Model, ABC):
  @abstractmethod
  def __init__(self, market: Market, **kwargs):
    """
    Abstract definition of a model with covariates.

    Args:
      `market`: a Market indexed by 'ID' and 'time'
      **kwargs:
      - `covar_vars` [str]: A required list of covariate variable names, each entry `covar_vars[i]` must be a xr.Variable in `market.xarray_ds`.
    """
    assert 'covar_vars' in kwargs, 'When creating a CovariateModel instance, you must specify a list of column names, `covar_vars`, that will be terms in the regression.'
    super().__init__(market, **kwargs)
    self.covar_vars = kwargs['covar_vars']
    self.thetas = np.zeros((len(self.covar_vars), ), dtype=np.float32)

  @abstractmethod
  def summarize_covariate_distributions(self):
    pass


class IndependentCovariateModel(IndependentModel, CovariateModel):
  def __init__(self, market):
    super().__init__(market)
    super(CovariateModel, self).__init__(market)

  def estimate_parameters(self, **kwargs):
    raise NotImplementedError("indep covar model")

  def simulate(self, dates, num_sims, **kwargs):
    raise NotImplementedError("indep covar model")

  def summarize_covariate_distributions(self):
    raise NotImplementedError("indep covar model")


class DependentCovariateModel(DependentModel, CovariateModel):
  def __init__(self, market, **kwargs):
    super().__init__(market, **kwargs)

  def estimate_parameters(self, **kwargs):
    self.drifts, self.Sigma, self.thetas = get_dep_cov_MLE_params(self.market, self.covar_vars, **kwargs)
    self.vols = np.linalg.norm(self.Sigma, axis=0, keepdims=True)

  def simulate(self, S_0, dates, num_sims, **kwargs):
    """
    simulate future market behavior

    Args:
      `S_0`: the market state at the start of the simulation, shaped (self.N_ASSETS,).
      `dates`: the dates to simulate, with dates[0] being the simulation's start time.
      `num_sims`: number of simulations.
      kwargs:
        `ds`: xarray.Dataset (optional) the timeseries dataset to use, defaults to `self.market.xarray_ds`.
        `avg_map`: dict{str:set(int)} (optional) a dictionary of covariate var names : set(time spans var was averaged over), defaults to `self.market.get_avg_vars()`.
        `add_BM`: bool (optional) indicating whether to add Brownian Motion (randomness) to simulation.

    Returns: a 3D numpy array [num_sims x self.N_ASSETS x len(dates)]
    """
    sim_res = np.zeros((num_sims, self.N_ASSETS, len(dates)))
    ts = (dates - dates.iloc[0]).dt.days

    ds = kwargs.get('ds', self.market.xarray_ds)
    t_start_idx = np.argmax(ds['time'].values == dates.iloc[0])
    if t_start_idx == -1:
      raise ValueError('dates[0] should have a frame in ds. Note, ds can also be provided as a kwarg')
    ds = ds.isel(time=slice(0, t_start_idx+1))

    covar_indices, _ = get_covar_indices_and_names(self.market.xarray_ds, self.covar_vars)

    for n_sim in range(num_sims):
      sim_res[n_sim, :, :] = sample_dep_cov_GBM(
        self.drifts, self.Sigma, self.thetas,
        S_0,
        ts[1:],
        ds, t_start_idx,
        self.market.derived_variables,
        covar_indices,
        add_BM=kwargs.get('add_BM', True)
      )

    return sim_res

  def summarize_covariate_distributions(self):
    for i, col_name in enumerate(self.covar_vars):
      print(col_name, format(self.thetas[i], '.4f'), sep='\t\t')