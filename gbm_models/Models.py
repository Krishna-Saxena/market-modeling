from abc import ABC, abstractmethod
from utils import *


class Model(ABC):
  def __init__(self, market):
    self.market = market
    self.N_ASSETS = len(self.market.assets)

  @abstractmethod
  def estimate_parameters(self):
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

  def estimate_parameters(self):
    raise NotImplementedError("IndependentModel.estimate_parameters()")

  def simulate(self, dates, num_sims, **kwargs):
    raise NotImplementedError("IndependentModel.simulate()")


class DependentModel(Model):
  def __init__(self, market):
    super().__init__(market)

    self.drifts = np.zeros((self.N_ASSETS,), dtype=np.float32)
    self.vols = np.zeros((self.N_ASSETS, self.N_ASSETS), dtype=np.float32)

  def estimate_parameters(self):
    raise NotImplementedError("DependentModel.estimate_parameters()")

  def simulate(self, dates, num_sims, **kwargs):
    raise NotImplementedError("DependentModel.simulate()")

  def get_correlation_mat(self):
    raise NotImplementedError("DependentModel.get_correlation_mat()")


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
