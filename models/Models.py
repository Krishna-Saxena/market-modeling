from abc import ABC, abstractmethod
from typing import Generator

import numpy as np
import pandas as pd
import xarray as xr

from markets.Markets import Market


class Model(ABC):
	@abstractmethod
	def fit_global_params(
		self, train_dss: [xr.Dataset], val_dss: [xr.Dataset] = None,
		train_epochs: int = 100, val_freq: int = None,
		**kwargs
	):
		"""
		Fits parameters for this model that are global across Markets.
		For example, covariate parameters are constant across all markets, but covariance matrices are individualized for each market.

		Args:
			train_dss: [xr.Dataset] a list of training datasets.
			val_dss: [xr.Dataset] (optional) a list of validation datasets.
			train_epochs: (optional) the number of training epochs, defaults to 100.
			val_freq: (optional) the number of update steps between each validation metric calculation, defaults to the number of minibatches in the train dataset.
			**kwargs: keyword arguments as defined by specific implementations of Model.
			- all kwargs to ~/modeling_utils.xr_utils.serve_xr_ds(), which will be used to batch each Dataset in train_dss and val_dss.

		Returns: A dict with information about the training curves. The learned parameters will be saved in **this Model's internal attributes.**
		"""
		pass

	@abstractmethod
	def fit_local_params(
		self, market: Market,
		train_ds: xr.Dataset = None, val_ds: xr.Dataset = None,
		train_epochs: int = 100, val_freq: int = None,
		**kwargs
	):
		"""
		Conducts a train, val, test split on `market`'s time axis and then learns parameters specific to a Market instance.
		This method should be called **after `fit_parameters()`** if this Model type has global parameters (e.g., covariate coefficients). Raises ValueError if this Model type has unfitted global parameters.
		For example, covariate coefficients are learned in `fit_parameters()` using all past data, and
		 drifts and covariances are optimized for the training data and the global covariate parameters are considered constant.

		Args:
			market: a `Market` instance. If `None`, then `train_ds` must be passed.
			train_ds: (optional) a xr.Dataset of training data. Must be indexed by 'ID' and 'time' with at least variable 'signal' like `Market.xarray_ds`. Ignored if `market` is not `None`.
			val_ds: (optional) a xr.Dataset of validation data. Same indexing, variable rules as `train_ds`. Ignored if `market` is not `None`.
			train_epochs: (optional) the number of training epochs, defaults to 100.
			val_freq: (optional) the number of update steps between each validation metric calculation, defaults to the number of minibatches in the train dataset.
			**kwargs: keyword arguments as defined by specific implementations of Model.
			- `val_size`, `test_size` (float, \\in (0, 1)): passed into `market.train_test_split()` to create training and validation sets.

		Returns: A tuple of parameters that are specific to this model and market and a dict with information about the training curves.
		"""
		pass

	@abstractmethod
	def simulate(self, market: Market, S_0, dates, num_sims, add_bm: bool = True, **kwargs):
		"""
		Simulate future market behavior.

		Args:
			market: a `Market` whose `signal` variable will be simulated.
			S_0: the market state at the start of the simulation, shaped (n_assets,)
			dates: the dates to simulate, with dates[0] being the simulation's start time
			num_sims: number of simulations
			add_bm (bool): optional flag indicating whether to add Brownian Motion (randomness) to simulation, defaults to True.
			**kwargs: optional keyword arguments. **Importantly,** learned local parameters as learned by `fit_dataset()`.

		Returns: a 3D numpy array [num_sims x n_assets x len(dates)]
		"""
		pass


class CovariateModel(Model, ABC):
	@abstractmethod
	def __init__(self, covar_vars: [str]):
		"""
		Abstract definition of a model with covariates.

		Args:
		  `market`: a Market indexed by 'ID' and 'time'
		  `covar_vars`: a list of covariate variable names
		"""
		if len(covar_vars) == 0:
			raise ValueError("'covar_vars' must contain at least one covariate variable. If you don't want covariates, use a DependentModel instead.")
		self.covar_vars = covar_vars
		self._thetas = np.NaN * np.ones((len(self.covar_vars),), dtype=np.float32)
		self._mean_covar_correction = np.NaN

	def summarize_covariate_distributions(self):
		"""
		A visual display of the distribution of each covariate parameter.
		"""
		return pd.DataFrame({
		  'variable': self.covar_vars,
		  'coefficients': self._thetas
		})