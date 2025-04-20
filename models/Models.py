from abc import ABC, abstractmethod
from typing import Generator

import xarray as xr

class Model(ABC):
  def __init__(self, **kwargs):
    pass

  @abstractmethod
  def fit_parameters(self, train_gen: Generator[xr.Dataset], val_ds: [xr.Dataset] = None, **kwargs):
    """
    Fits parameters for this model that are constant across all Markets.
    For example, covariate parameters are constant across all markets, but covariance matrices are individualized for each market.

    Args:
      train_gen: a generator that yields training data (see e.g., ~/modeling_utils.xr_utils.serve_xr_ds())
      val_ds: (optional) a generator that yields validation data
      **kwargs: keyword arguments as defined by specific implementations of Model.

    Returns: None. Modifies this Model's internal attributes.
    """
    pass

  @abstractmethod
  def model_market(self, train_gen: Generator[xr.Dataset], val_ds: [xr.Dataset] = None, **kwargs):
    """
    Fits parameters for this model that are individualized for this specific Market instance.
    **Generally, this method should be called after fit_parameters().**
    For example, covariate coefficients are learned in fit_parameters() using all past data, and
     drifts, covariances are learned for `market` given the historically-optimal covariate parameters.

    Args:
      train_gen: a generator that yields training data (see e.g., ~/modeling_utils.xr_utils.serve_xr_ds())
      val_ds: (optional) a generator that yields validation data
      **kwargs: keyword arguments as defined by specific implementations of Model.

    Returns: A tuple of parameters that are specific to this model.
    """
    pass

  @abstractmethod
  def simulate(self, S_0, dates, num_sims, **kwargs):
    """
    simulate future market behavior

    Args:
      S_0: the market state at the start of the simulation, shaped (self.N_ASSETS,)
      dates: the dates to simulate, with dates[0] being the simulation's start time
      num_sims: number of simulations
      kwargs: optional keyword arguments
        add_BM: bool indicating whether to add Brownian Motion (randomness) to simulation

    Returns: a 3D numpy array [num_sims x self.N_ASSETS x len(dates)]
    """
    pass