import numpy as np
import xarray as xr
import torch

from markets.Markets import Market
from models.Models import Model, CovariateModel
from models.geom_growth_models.gbm_modeling_utils import _get_nll_loss_dep, get_indep_MLE_params, sample_indep_GBM, \
  get_dep_MLE_params, sample_dep_GBM, learn_dep_cov_MLE_theta, sample_dep_cov_GBM, fit_dep_cov
from modeling_utils.xr_utils import get_dX_dt_timestamps
from modeling_utils.torch_utils import triu_mask_gen_fn, make_Tensor, detach
from metrics.EvalMetrics import mae


class IndependentModel(Model):
  def fit_global_params(
    self, train_dss: [xr.Dataset], val_dss: [xr.Dataset] = None,
    train_epochs: int = 100, val_freq: int = None,
    **kwargs
  ):
    """
    Since there are no global parameters for `IndependentModel`s, this is a no-op.
    """
    pass

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
      market: a `Market` instance. If `None`, then `train_gen` must be passed.
      train_ds: (optional) a xr.Dataset of training data. Must be indexed by 'ID' and 'time' with at least variable 'signal' like `Market.xarray_ds`. Ignored if `market` is not `None`.
      val_ds: (optional) a xr.Dataset of validation data. Same indexing, variable rules as `train_ds`. Ignored if `market` is not `None`.
      train_epochs: (**Ignored**) the number of training epochs, defaults to 100.
      val_freq: (**Ignored**) the number of update steps between each validation metric calculation, defaults to the number of minibatches in the train dataset.
      **kwargs: keyword arguments as defined by specific implementations of Model.

    Returns: (drifts, vols), {'train_loss': [NLL], 'val_loss': [NLL], 'train_MAE': [MAE], 'val_MAE': [MAE]}.
    """
    if market:
      tr_ds, val_ds, _ = market.train_test_split(**kwargs)
    else:
      tr_ds = train_ds

    n_assets = len(tr_ds.ID)
    drifts = np.zeros((n_assets,), dtype=np.float32)
    vols = np.zeros((n_assets,), dtype=np.float32)

    for i in range(n_assets):
      drifts[i], vols[i] = get_indep_MLE_params(tr_ds.isel(ID=i))

    dX_train, dt_train, timestamps_train = get_dX_dt_timestamps(tr_ds.to_dataarray())
    if val_ds is not None:
      dX_val, dt_val, timestamps_val = get_dX_dt_timestamps(val_ds.to_dataarray())
    Sigma = np.diag(vols)

    with torch.no_grad():
      train_hist_info = {
        'epochs': [1],
        'train_loss': [_get_nll_loss_dep(make_Tensor(dX_train), make_Tensor(dt_train), make_Tensor(drifts), make_Tensor(Sigma)).item()],
        'val_loss': [
          _get_nll_loss_dep(make_Tensor(dX_val), make_Tensor(dt_val), make_Tensor(drifts), make_Tensor(Sigma)).item() if
            val_ds is not None else np.nan
        ],
        'train_MAE': [
          mae(
            tr_ds['signal'].to_numpy().T[:, 1:],
            sample_dep_GBM(drifts, Sigma, tr_ds['signal'].isel(time=0).to_numpy(), timestamps_train, False)[:, 1:]
          )
        ],
        'val_MAE': [
          mae(
            val_ds['signal'].to_numpy().T[:, 1:],
            sample_dep_GBM(drifts, Sigma, val_ds['signal'].isel(time=0).to_numpy(), timestamps_val, False)[:,1:]
          ) if val_ds is not None else np.nan
        ],
      }
    return (drifts, vols), train_hist_info

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
      - mu: a vector of shape (n_assets,) with drifts for each asset.
      - Sigma: a vector of shape (n_assets,) with volatilities for each asset.

    Returns: a 3D numpy array [num_sims x n_assets x len(dates)]
    """

    if 'mu' not in kwargs:
      raise ValueError("'mu' must be passed as a kwarg. Generally, drifts is returned by self.fit_local_params().")
    if 'vols' not in kwargs:
      raise ValueError("'Sigma' must be passed as a kwarg. Generally, drifts is returned by self.fit_local_params().")
    n_assets = S_0.shape[0]
    sim_res = np.zeros((num_sims, n_assets, len(dates)))
    ts = (dates - dates.iloc[0]).dt.days

    for n_sim in range(num_sims):
      for i in range(n_assets):
        sim_res[n_sim, i, :] = sample_indep_GBM(
          kwargs['mu'][i],
          kwargs['vols'][i],
          S_0[i],
          ts.values[1:],
          kwargs.get('add_BM', True)
        )
    return sim_res


class DependentModel(Model):
  def fit_global_params(
    self, train_dss: [xr.Dataset], val_dss: [xr.Dataset] = None,
    train_epochs: int = 100, val_freq: int = None,
    **kwargs
  ):
    """
    Since there are no global parameters for `DependentModel`s, this is a no-op.
    """
    pass

  def fit_local_params(
    self, market: Market,
    train_ds: xr.Dataset = None, val_ds: xr.Dataset = None,
    train_epochs: int = 100, val_freq: int = 1,
    **kwargs
  ):
    """
    Conducts a train, val, test split on `market`'s time axis and then learns parameters specific to a Market instance.
    This method should be called **after `fit_parameters()`** if this Model type has global parameters (e.g., covariate coefficients). Raises ValueError if this Model type has unfitted global parameters.
    For example, covariate coefficients are learned in `fit_parameters()` using all past data, and
     drifts and covariances are optimized for the training data and the global covariate parameters are considered constant.

    Args:
      market: a `Market` instance. If `None`, then `train_gen` must be passed.
      train_ds: (optional) a xr.Dataset of training data. Must be indexed by 'ID' and 'time' with at least variable 'signal' like `Market.xarray_ds`. Ignored if `market` is not `None`.
      val_ds: (optional) a xr.Dataset of validation data. Same indexing, variable rules as `train_ds`. Ignored if `market` is not `None`.
      train_epochs: (optional) the number of training epochs, defaults to 100.
      val_freq: (optional) the number of epochs between each validation metric calculation, defaults to 1.
      **kwargs: keyword arguments as defined by specific implementations of Model.
      - `val_size`, `test_size` (float, \\in (0, 1)): passed into `market.train_test_split()` to create training and validation sets.

    Returns: A tuple of parameters that are specific to this model and market and a dict with information about the training curves.
    """
    if market:
      tr_ds, val_ds, _ = market.train_test_split(**kwargs)
    else:
      tr_ds = train_ds

    return get_dep_MLE_params(tr_ds, val_ds, n_epochs=train_epochs, **kwargs)

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
    if 'mu' not in kwargs:
      raise ValueError("'mu' must be passed as a kwarg. Generally, `drifts` is returned by `self.fit_local_params()`.")
    if 'Sigma' not in kwargs:
      raise ValueError("'Sigma' must be passed as a kwarg. Generally, `Sigma=diag(vols)` where `vols` was returned by `self.fit_local_params()`.")

    n_assets = S_0.shape[0]
    sim_res = np.zeros((num_sims, n_assets, len(dates)))
    ts = (dates - dates.iloc[0]).dt.days.values

    for n_sim in range(num_sims):
      sim_res[n_sim, :, :] = sample_dep_GBM(
        kwargs['mu'], kwargs['Sigma'],
        S_0,
        ts[1:],
        add_BM=kwargs.get('add_BM', True)
      )

    return sim_res


class DependentCovariateModel(DependentModel, CovariateModel):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)

  def fit_global_params(
    self, train_dss: [xr.Dataset], val_dss: [xr.Dataset] = None,
    train_epochs: int = 100, val_freq: int = 1,
    **kwargs
  ):
    """
    Fits parameters for this model that are global across Markets.
    For example, covariate parameters are constant across all markets, but covariance matrices are individualized for each market.

    Args:
      train_dss: [xr.Dataset] a list of training datasets.
      val_dss: [xr.Dataset] (optional) a list of validation datasets.
      train_epochs: (optional) the number of training epochs, defaults to 100.
      val_freq: (optional) the number of epochs between each validation metric calculation, defaults to 1.
      **kwargs: keyword arguments as defined by specific implementations of Model.
      - var_types {str : VarType}
      - der_var_specs [DerivedVarSpec]
      - all kwargs to ~/modeling_utils.xr_utils.serve_xr_ds(), which will be used to batch each Dataset in train_dss and val_dss.

    Returns: A dict with information about the training curves. The learned parameters will be saved in **this Model's internal attributes.**
    """
    theta, train_hist_info = learn_dep_cov_MLE_theta(
      self.covar_vars, kwargs.pop('var_types'), kwargs.pop('der_var_specs'), train_dss, val_dss,
      n_epochs=train_epochs, val_freq = val_freq,
      save_hist=kwargs.pop('save_hist', False),
      **kwargs
    )
    self._thetas = theta
    return theta, train_hist_info

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
      market: a `Market` instance. If `None`, then `train_gen` must be passed.
      train_ds: (optional) a xr.Dataset of training data. Must be indexed by 'ID' and 'time' with at least variable 'signal' like `Market.xarray_ds`. Ignored if `market` is not `None`.
      val_ds: (optional) a xr.Dataset of validation data. Same indexing, variable rules as `train_ds`. Ignored if `market` is not `None`.
      train_epochs: (optional) the number of training epochs, defaults to 100.
      val_freq: (optional) the number of update steps between each validation metric calculation, defaults to the number of minibatches in the train dataset.
      **kwargs: keyword arguments as defined by specific implementations of Model.

    Returns: A tuple of parameters that are specific to this model and market and a dict with information about the training curves.
    """
    if market:
      tr_ds, _, _ = market.train_test_split()
    else:
      tr_ds = train_ds

    mu, A = fit_dep_cov(
      tr_ds, make_Tensor(kwargs.pop('theta', self._thetas)), kwargs.pop('covar_vars'),
      triu_mask_gen_fn, l1_penalty=0.,
      steps_per_batch=kwargs.pop('steps_per_batch', 50),
      mu=None, A=None, train_theta=False,
      **kwargs
    )
    return detach(mu), detach(A@A.T)


  def simulate(self, market, S_0, dates, num_sims, **kwargs):
    """
    simulate future market behavior

    Args:
      market: a Market whose signal timeseries will be simulated.
      S_0: the market state at the start of the simulation, shaped (market.get_num_assets(),).
      dates: the dates to simulate, with dates[0] being the simulation's start time.
      num_sims: number of simulations.
      kwargs:
      - mu, Sigma, theta
      - ds: xarray.Dataset (optional) the timeseries dataset to use, defaults to `self.market.xarray_ds`.
      - avg_map: dict{str:set(int)} (optional) a dictionary of covariate var names : set(time spans var was averaged over), defaults to `self.market.get_avg_vars()`.
      - add_BM: bool (optional) indicating whether to add Brownian Motion (randomness) to simulation.

    Returns: a 3D numpy array [num_sims x market.get_num_assets() x len(dates)]
    """
    sim_res = np.zeros((num_sims, market.get_num_assets(), len(dates)))
    ts = (dates - dates.iloc[0]).dt.days.values

    ds = kwargs.get('ds', market.xarray_ds)
    t_start_idx = np.argmax(ds['time'].values == dates.iloc[0])
    if t_start_idx == -1:
      raise ValueError('dates[0] should have a frame in ds. Note, ds can also be provided as a kwarg')
    # ds = ds.isel(time=slice(0, t_start_idx + 1))

    for n_sim in range(num_sims):
      sim_res[n_sim, :, :] = sample_dep_cov_GBM(
        kwargs['mu'], kwargs['Sigma'], kwargs['theta'],
        S_0,
        ts[1:],
        ds, t_start_idx,
        market.var_types,
	      market.derived_variables,
	      self.covar_vars,
        add_BM=kwargs.get('add_BM', True)
      )

    return sim_res
