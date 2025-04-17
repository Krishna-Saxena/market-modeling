from logging import warning
from abc import ABC, abstractmethod
from collections import defaultdict
from copy import deepcopy

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from metrics.Metrics import VarType
from assets.Assets import Asset
from modeling_utils.py_utils import apply_one_encode, apply_many_encode, zeros_like_nan, get_dataset_only_numeric_vars, \
  last_index_lt_1D, make_avg_col_name, ffill_df, bfill_df
from modeling_utils.xr_utils import ffill_via_pandas, bfill_via_pandas
from models.gbm_models.gbm_modeling_utils import ffill_1D_GBM, bfill_1D_GBM


class Market(ABC):
  def _collect_levels(self, sep=','):
    """
    Utility method to collect the levels of CATEGORICAL and MULTI_CATEGORICAL variables in a Market.

    Args:
      `sep`: the separator between levels in a MULTI_CATEGORICAL variable's string value.

    Returns: None. Populates self.levels as a dict {var_name : {possible value : int ID}}.
    """
    for asset in self.assets:
      for metric_name, metric_type in asset.var_types.items():
        if metric_type == VarType.CATEGORICAL:
          self.levels[metric_name] = self.levels.get(metric_name, set())
          self.levels[metric_name].update(asset.ts_df[metric_name].unique().tolist())
        elif metric_type == VarType.MULTI_CATEGORICAL:
          self.levels[metric_name] = self.levels.get(metric_name, set())
          for vals in asset.ts_df[metric_name].unique():
            self.levels[metric_name].update(vals.split(sep))
    for metric_name in self.levels:
      self.levels[metric_name] = {l:i for i, l in enumerate(list(self.levels[metric_name]))}

  @abstractmethod
  def __init__(self, assets: [Asset], market_name: str):
    self.assets = sorted(assets, key=lambda asset: asset.asset_id)
    self.market_name = market_name
    self.levels = {}
    self.num_covars = len(assets[0].var_types)-1 # -1 to account for `signal` not being a covar

    # dict of var_name : [time spans var_name was avg'd over]
    self._avg_vars = defaultdict(set)

  @abstractmethod
  def align_timeseries(self, impute=False, **kwargs):
    """
    Modifies the internal representation so that all timestamps have non-NaN feature vectors.

    Args:
      `impute`: whether to impute missing values (using a stochastic method) or not.
      **kwargs: keyword arguments to help with imputation.
      - cols_to_exclude: cols whose NaN values will not be filed by directly copying its neighbors' values.
      - col_to_fill: a column in whose NaN values will be filled by 1D GBM simulations, 'signal' by default.
      - num_sims: number of 1D GBM simulations for each NaN value, 100 by default.

    Returns: None, modifies the Market's internal representation.
    """
    pass

  @abstractmethod
  def display_market(self, **kwargs):
    pass

  @abstractmethod
  def encode_indicators(self, drop=True, sep=','):
    """
    Collects the levels of CATEGORICAL and MULTI_CATEGORICAL covariates and updates the Market's internal representation
     so by adding indicator columns for each covariate's levels.

    Args:
      `drop`: whether to drop the original CATEGORICAL and MULTI_CATEGORICAL variables from the internal representation.
      `sep`: the separator between levels in a MULTI_CATEGORICAL variable's string value.

    Returns: `None`. updates self.levels, self.num_covars, and the Market's internal representation.
    """
    self._collect_levels(sep)
    if len(self.levels) == 0:
      warning('No CATEGORICAL or MULTI_CATEGORICAL variables tagged for this market. market.update_levels() is a no-op')

  @abstractmethod
  def add_avg_var(self, var_name: str, time_span: int):
    """
    Creates a new variable from an exponentially weighted moving average of a variable in the Market.

    Args:
      `var_name`: the variable whose average will be added, must be a VarType.BINARY or VarType.QUANTITATIVE.
      `time_span`: controls the decay parameter of the EWMA. Described in detail at https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.ewm.html.

    Returns: the name of the variable added. updates self.num_covars, and the Market's internal representation.
    """
    pass

  def get_avg_vars(self):
    return deepcopy(self._avg_vars)

  @abstractmethod
  def get_market_state_before_date(self, date):
    """
    Get the market state before a given date.

    Args:
      `date`: the state will have a timestep less than this date.

    Returns: the date(s) before date selected from, the market state.
    """
    pass


class BaseMarket(Market):
  def __init__(self, assets, market_name=''):
    super().__init__(assets, market_name)

  def align_timeseries(self, impute=False, **kwargs):
    if impute:
      supserset_times = set(self.assets[0].ts_df['time'].values)
      for asset in self.assets[1:]:
        supserset_times = supserset_times.union(asset.ts_df['time'].values)
      superset_times_df = pd.DataFrame({'time': pd.to_datetime(supserset_times)})
      superset_times_df.sort_values('time', inplace=True)

      for asset in self.assets:
        comb_df = pd.concat((asset.ts_df, superset_times_df))
        asset.ts_df = comb_df.groupby(['time'], sort=True, as_index=False).first().reset_index()
        asset.ts_df = ffill_1D_GBM(asset.ts_df, **kwargs)
        asset.ts_df = bfill_1D_GBM(asset.ts_df, **kwargs)
    else:
      common_times = set(self.assets[0].ts_df['time'].values)
      for asset in self.assets[1:]:
        common_times = common_times.intersection(asset.ts_df['time'].values)

      for asset in self.assets:
        asset.ts_df = asset.ts_df[asset.ts_df.time.isin(common_times)]

    # ffill and bfill covariates
    for asset in self.assets:
      asset.ts_df = ffill_df(asset.ts_df, **kwargs)
      asset.ts_df = bfill_df(asset.ts_df, **kwargs)

  def display_market(self, **kwargs):
    """
    Print the market to console.

    Args:
      **kwargs: `full_timeseries`: bool, whether to display each asset's full timeseries.
    """

    print(f'Start Market {self.market_name}')
    for asset in self.assets:
      if kwargs.get('full_timeseries', False):
        print(asset.ts_df)
      else:
        print(asset.ts_df.head(2))
        print('...')
        print(asset.ts_df.tail(2))
      print()
    print(f'End Market {self.market_name}')

  def encode_indicators(self, drop=True, sep=','):
    super().encode_indicators(drop, sep)
    for var_name, level_id_map in self.levels.items():
      id_level_map = {lvl:_id for _id, lvl in level_id_map.items()}
      new_cols = [f'{var_name}_{id_level_map[i]}' for i in range(len(level_id_map))]
      new_cols_added = False

      for asset in self.assets:
        if var_name in asset.ts_df.columns:
          new_cols_added = True
          for col in new_cols:
            asset.var_types[col] = VarType.BINARY
          # add columns for levels to each asset's dataframe
          for col in new_cols:
            asset.ts_df[col] = asset.ts_df[var_name].isna().astype(np.float16)
            asset.ts_df[col] = asset.ts_df[col].apply(lambda x: 0 if x == 0 else np.NaN)
          # fill in the ones
          if asset.var_types[var_name] == VarType.CATEGORICAL:
            asset.ts_df = asset.ts_df.apply(
              apply_one_encode, axis=1, var_name=var_name, zeros_cols=new_cols, level_id_map=level_id_map
            )
          elif asset.var_types[var_name] == VarType.MULTI_CATEGORICAL:
            asset.ts_df = asset.ts_df.apply(
              apply_many_encode, axis=1, sep=sep, var_name=var_name, zeros_cols=new_cols, level_id_map=level_id_map
            )
          # drop old column
          if drop:
            asset.ts_df.drop(columns=var_name, inplace=True)
            del asset.var_types[var_name]

      if new_cols_added:
        self.num_covars += len(new_cols)
        if drop:
          self.num_covars -= 1

  def get_market_state_before_date(self, date):
    """
    Get the market state before a given date.

    Args:
      `date`: the state will have a timestep less than this date.

    Returns: a list of pandas Series, with the i^th series being the i^th asset's state.
    """
    times = []
    states = []

    for i, asset in enumerate(self.assets):
      t_index = last_index_lt_1D(asset.ts_df['time'], date)
      times.append(asset.ts_df['time'].iloc[t_index])
      states.append(asset.ts_df.iloc[t_index, :])

    return times, states

  def add_avg_var(self, var_name: str, time_span: int):
    for asset in self.assets:
      if var_name not in asset.var_types:
        raise KeyError(f"Variable {var_name} must be present in all of this Market's assets")
      elif asset.var_types[var_name] != VarType.BINARY and asset.var_types[var_name] != VarType.QUANTITATIVE:
        raise KeyError(f"Variable {var_name} must be a VarType.BINARY or VarType.QUANTITATIVE in all of this Market's assets")

    new_var_name = make_avg_col_name(var_name, time_span)
    new_var_added = False
    for asset in self.assets:
      new_var_added = new_var_added or new_var_name in asset.var_types
      asset.var_types[new_var_name] = VarType.QUANTITATIVE
      asset.ts_df[new_var_name] = asset.ts_df[var_name].ewm(span=time_span, times=asset.ts_df['time']).mean()
      asset.ts_df[new_var_name] = asset.ts_df[new_var_name].ffill(dim='time')
    if new_var_added:
      self.num_covars += 1
      self._avg_vars[var_name].add(time_span)
    return new_var_name


class XarrayMarket(Market):
  def __init__(self, assets, market_name=''):
    super().__init__(assets, market_name)

    df_combined = pd.concat([asset.ts_df.set_index(['time', 'ID']) for asset in assets], sort=False)
    self.xarray_ds = xr.Dataset.from_dataframe(df_combined)

  def get_dataarray(self):
    return self.xarray_ds.to_dataarray()

  def align_timeseries(self, impute=False, **kwargs):
    col_to_fill = kwargs.get('col_to_fill', 'signal')
    if impute:
      for i in range(len(self.assets)):
        filled_df = ffill_1D_GBM(self.xarray_ds.isel(ID=i)[col_to_fill].to_dataframe().reset_index(), **kwargs)
        filled_df = bfill_1D_GBM(filled_df, **kwargs)
        self.xarray_ds.isel(ID=i)[col_to_fill].data[:] = filled_df[col_to_fill].values

    else:
      self.xarray_ds = self.xarray_ds.dropna(dim='time', subset=[col_to_fill])

    self.xarray_ds = ffill_via_pandas(self.xarray_ds)
    self.xarray_ds = bfill_via_pandas(self.xarray_ds)

  def display_market(self, **kwargs):
    """
    Print the market to console.

    Args:
      **kwargs: `full_timeseries`: bool, whether to display the full timeseries.
                `show_indicators`: bool, whether to display variables of VarType.BINARY.
    """

    xarr = get_dataset_only_numeric_vars(self.xarray_ds).to_dataarray()

    if kwargs.get('full_timeseries', False):
      xarr = xr.concat((xarr.isel(time=slice(0, 2)), xarr.isel(time=slice(-2, None))), dim='time')
    if not kwargs.get('show_indicators', True):
      for var_name, var_type in self.assets[0].var_types.items():
        if var_type == VarType.BINARY:
          xarr = xarr.drop_sel(variable=var_name)

    # x: time, y: vars, z: assets
    x_ax = xarr.time.values.astype(np.int64)
    y_ax = np.arange(xarr.variable.shape[0])
    z_ax = np.arange(xarr.ID.shape[0])

    xx, yy, zz = np.meshgrid(x_ax, y_ax, z_ax)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlabel('Time', c='b')
    ax.set_xticks(xarr.time)
    # ax.set_yticks(pd.date_range(
    #   start=xarr.time.values[0],
    #   end=xarr.time.values[-1],
    #   freq="W")
    # )
    ax.set_ylabel('Measure', c='b')
    ax.set_yticks(y_ax, xarr.indexes['variable'].values, c='m')
    ax.set_zlabel('Asset ID', c='b')
    ax.set_zticks(z_ax, [_id[:10] for _id in xarr.ID.values], c='m')
    ax.set_title(f'3D Visualization of Market {self.market_name}')

    cbar = ax.scatter(
      xx, yy, zz,
      c=xarr.to_numpy()+1e-2, cmap='plasma', norm=LogNorm()
    )

    fig.colorbar(cbar, ax=ax, shrink=0.5, aspect=5, pad=0.1, label='')
    # plt.gcf().autofmt_xdate()
    plt.show()

  def encode_indicators(self, drop=True, sep=','):
    super().encode_indicators(drop, sep)
    for var_name, level_id_map in self.levels.items():
      id_level_map = {lvl:_id for _id, lvl in level_id_map.items()}
      new_cols = [f'{var_name}_{id_level_map[i]}' for i in range(len(level_id_map))]

      if var_name in self.xarray_ds.variables:
        for asset in self.assets:
          for col in new_cols:
            asset.var_types[col] = VarType.BINARY

        self.xarray_ds = self.xarray_ds.assign({col:zeros_like_nan(self.xarray_ds[var_name]) for col in new_cols})
        # TODO: read https://stackoverflow.com/a/71425308 and use apply_ufunc() instead of to_dataframe()+apply()
        ext_df = self.xarray_ds.to_dataframe()
        if self.assets[0].var_types[var_name] == VarType.CATEGORICAL:
          ext_df = ext_df.apply(apply_one_encode, axis=1, var_name=var_name, zeros_cols=new_cols, level_id_map=level_id_map)
        elif self.assets[0].var_types[var_name] == VarType.MULTI_CATEGORICAL:
          ext_df = ext_df.apply(apply_many_encode, axis=1, var_name=var_name, zeros_cols=new_cols,level_id_map=level_id_map)
        self.xarray_ds = xr.Dataset.from_dataframe(ext_df)

        self.num_covars += len(new_cols)
        if drop:
          # drop old column
          self.xarray_ds = self.xarray_ds.drop_vars(var_name)
          for asset in self.assets:
            del asset.var_types[var_name]
          self.num_covars -= 1

  def get_market_state_before_date(self, date):
    """
    Get the market state before a given date.

    Args:
      `date`: the state will have a timestep less than this date.

    Returns: a list of one time, an xarray.XArray of the market's state at that time.
    """
    xarr = self.get_dataarray()
    market_t_index = last_index_lt_1D(xarr.time.values, date)

    return [xarr.time.values[market_t_index]], xarr.isel(time=market_t_index)

  def add_avg_var(self, var_name: str, time_span: int):
    if self.xarray_ds[var_name].dtype == 'O':
      raise KeyError(f"Variable {var_name} must be numeric")

    new_var_name = make_avg_col_name(var_name, time_span)
    if new_var_name not in self.xarray_ds:
      self.xarray_ds[new_var_name] = self.xarray_ds[var_name].rolling_exp(
        window={'time': 7}, window_type='span'
      ).mean()
      self.xarray_ds[new_var_name] = self.xarray_ds[new_var_name].ffil(dim='time')
      self.num_covars += 1
      self._avg_vars[var_name].add(time_span)
    return new_var_name