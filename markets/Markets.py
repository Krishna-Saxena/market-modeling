from logging import warning
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from markets.DerivedVariables import DerivedVariable
from metrics.Metrics import VarType
from assets.Assets import Asset
from modeling_utils.py_utils import make_factor_col_name, apply_one_encode, apply_many_encode, last_index_lt_1D, drop_indices_from_array
from modeling_utils.xr_utils import ffill_via_pandas, bfill_via_pandas, zeros_like_nan, get_dataset_only_numeric_vars, drop_if_var_na
from models.geom_growth_models.gbm_modeling_utils import ffill_1D_GBM, bfill_1D_GBM
from models.arith_growth_models.arith_modeling_utils import ffill_1D_arith, bfill_1D_arith


class Market:
  def _collect_levels(self, sep=','):
    """
    Utility method to collect the levels of CATEGORICAL and MULTI_CATEGORICAL variables in a Market.

    Args:
      `sep`: the separator between levels in a MULTI_CATEGORICAL variable's string value.

    Returns: None. Populates self.levels as a dict {var_name : {possible value : int ID}}.
    """
    levels = {}
    for metric_name, metric_type in self.var_types.items():
      if metric_type == VarType.CATEGORICAL:
        levels[metric_name] = pd.unique(self.xarray_ds[metric_name].data.reshape((-1,))).tolist()
      elif metric_type == VarType.MULTI_CATEGORICAL:
        levels[metric_name] = set()
        for vals in pd.unique(self.xarray_ds[metric_name].data.reshape((-1,))):
          levels[metric_name].update(vals.split(sep))
        levels[metric_name] = list(levels[metric_name])
    return levels

  def __init__(self, assets: [Asset], market_name: str = '', var_types=None):
    """
    Create a new Market.

    Args:
      assets: a list of Assets.
      market_name: name of this market.
      var_types: a dict of {var_name : Metrics.VarType}.
    """
    assets = sorted(assets, key=lambda asset: asset.asset_id)
    self.id_name_map = {asset.asset_id : asset.name for asset in assets}
    self.var_types = assets[0].var_types if var_types is None else var_types
    self.market_name = market_name

    df_combined = pd.concat([asset.ts_df.set_index(['time', 'ID']) for asset in assets], sort=False)
    self.xarray_ds = xr.Dataset.from_dataframe(df_combined)

    self.derived_variables = {}

  def get_num_assets(self) -> int:
    return len(self.xarray_ds.ID)

  def get_dataarray(self):
    return self.xarray_ds.to_dataarray()

  def remove_sparse_assets(self, nan_threshold=2/3):
    """
    Drops assets with NaN frequency more than `na_threshold` `var_name`.

    Args:
      nan_threshold: the proportion of NaN values in var_name that are acceptable, defaults to 2/3.
    """
    self.xarray_ds, _ = drop_if_var_na(self.xarray_ds, 'signal', nan_threshold)


  def align_timeseries(self, impute=False, use_geom_model=True, **kwargs):
    """
    Align the Market's internal representation such that there are no missing values in any asset's variables.

    Args:
      impute (bool): whether to impute missing values in the signal column (or remove timeframes where any asset is missing values in the signal column).
      use_geom_model (bool): whether to use an Independent Geometric Model or Independent Arithmetic Model to fill missing signal values.
      **kwargs: Optional keyword arguments.
      - col_to_fill (str): the name of the column to fill in the missing values, defaults to 'signal'.
    """
    col_to_fill = kwargs.get('col_to_fill', 'signal')
    if impute:
      for i in range(self.get_num_assets()):
        if use_geom_model:
          filled_df = ffill_1D_GBM(self.xarray_ds.isel(ID=i)[col_to_fill].to_dataframe().reset_index(), **kwargs)
          filled_df = bfill_1D_GBM(filled_df, **kwargs)
        else:
          filled_df = ffill_1D_arith(self.xarray_ds.isel(ID=i)[col_to_fill].to_dataframe().reset_index(), **kwargs)
          filled_df = bfill_1D_arith(filled_df, **kwargs)
        self.xarray_ds.isel(ID=i)[col_to_fill].data[:] = filled_df[col_to_fill].values

    else:
      self.xarray_ds = self.xarray_ds.dropna(dim='time', subset=[col_to_fill])

    self.xarray_ds = ffill_via_pandas(self.xarray_ds)
    self.xarray_ds = bfill_via_pandas(self.xarray_ds)

  def display_market(self, **kwargs):
    """
    Displays the market's timeseries as a 3D plt figure.

    Args:
      **kwargs:
      - `full_timeseries` (bool): whether to display the full timeseries, defaults to False.
      - `show_indicators` (bool): whether to display variables of VarType.BINARY, defaults to True.
    """

    xarr = get_dataset_only_numeric_vars(self.xarray_ds).to_dataarray()

    if kwargs.get('full_timeseries', False):
      xarr = xr.concat((xarr.isel(time=slice(0, 2)), xarr.isel(time=slice(-2, None))), dim='time')
    if not kwargs.get('show_indicators', True):
      indic_cols = [var_name for var_name, var_type in self.var_types.items() if var_type == VarType.BINARY]
      xarr = xarr.drop_sel(variable=indic_cols)

    # x: time, y: vars, z: assets
    x_ax = xarr.time.values.astype(np.int64)
    y_ax = np.arange(xarr.variable.shape[0])
    z_ax = np.arange(xarr.ID.shape[0])

    xx, yy, zz = np.meshgrid(x_ax, y_ax, z_ax)

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlabel('Time', c='b')
    ax.set_xticks(xarr.time.values)
    ax.set_ylabel('Measure', c='b')
    ax.set_yticks(y_ax, xarr.indexes['variable'].values, c='m', ha='left')
    ax.set_zlabel('Asset ID', c='b')
    ax.set_zticks(z_ax, [_id[:10] for _id in xarr.ID.values], c='m')
    ax.set_title(f'3D Visualization of Market {self.market_name}')

    cbar = ax.scatter(
      xx, yy, zz,
      c=xarr.to_numpy() + 1e-2, alpha=0.25,
      cmap='plasma', norm=LogNorm()
    )

    fig.colorbar(cbar, ax=ax, shrink=0.5, aspect=10, pad=0.1, label='')
    plt.gcf().autofmt_xdate()
    plt.show()

  def encode_indicators(self, drop=False, sep=','):
    """
    Encode each VarType.CATEGORICAL and VarType.MULTI_CATEGORICAL variable into VarType.BINARY indicators.

    Args:
      drop (bool): whether to drop any of the VarType.CATEGORICAL and VarType.MULTI_CATEGORICAL variables present at the start of the operation, default is False.
      sep: the separator between levels in a MULTI_CATEGORICAL variable's string value.

    Returns: None, updates the internal data representation.
    """
    levels = self._collect_levels(sep)
    if len(levels) == 0:
      warning('No CATEGORICAL or MULTI_CATEGORICAL variables tagged for this market. market.encode_indicators() is a no-op')

    for var_name, level_id_list in levels.items():
      level_id_map = {lvl: idx for idx, lvl in enumerate(level_id_list)}
      idx_level_map = {lvl: idx for idx, lvl in level_id_map.items()}
      new_cols = [f'{var_name}_{idx_level_map[i]}' for i in range(len(level_id_map))]

      for col in new_cols:
        self.var_types[col] = VarType.BINARY

      self.xarray_ds = self.xarray_ds.assign({col: zeros_like_nan(self.xarray_ds[var_name]) for col in new_cols})
    # TODO: read https://stackoverflow.com/a/71425308 and use apply_ufunc() instead of to_dataframe()+apply()
    ext_df = self.xarray_ds.to_dataframe()
    for var_name, level_id_list in levels.items():
      level_id_map = {lvl: idx for idx, lvl in enumerate(level_id_list)}
      id_level_map = {lvl: _id for _id, lvl in level_id_map.items()}
      new_cols = [make_factor_col_name(var_name, id_level_map[i]) for i in range(len(level_id_map))]
      if self.var_types[var_name] == VarType.CATEGORICAL:
        ext_df = ext_df.apply(apply_one_encode, axis=1, var_name=var_name, zeros_cols=new_cols,
                              level_id_map=level_id_map)
      elif self.var_types[var_name] == VarType.MULTI_CATEGORICAL:
        ext_df = ext_df.apply(apply_many_encode, axis=1, var_name=var_name, zeros_cols=new_cols,
                              level_id_map=level_id_map)
      self.xarray_ds = xr.Dataset.from_dataframe(ext_df)

      if drop:
        # drop old column
        self.xarray_ds = self.xarray_ds.drop_vars(var_name)
        del self.var_types[var_name]

  def get_market_state_before_date(self, date: datetime) -> (datetime, xr.DataArray):
    """
    Get the market state before a given date.

    Args:
      `date`: the state will have a timestep less than this date.

    Returns: the first timestep in the market before `date`, a xr.DataArray of the market's state at that time.
    """
    xarr = self.get_dataarray()
    market_t_index = last_index_lt_1D(xarr.time.values, date)

    return xarr.time.values[market_t_index], xarr.isel(time=market_t_index)

  def register_derived_var(self, der_var_spec: DerivedVariable):
    # save `der_var_spec` as the spec associated to this operation
    self.derived_variables[der_var_spec.op_name] = der_var_spec
    # save the types of the generated columns
    self.var_types.update(der_var_spec.gen_var_types)
    # initialize the new columns
    self.xarray_ds = der_var_spec.init_var(self.xarray_ds)

  def get_quant_covars(self):
    return [n for n, t in self.var_types.items() if t in {VarType.QUANTITATIVE, VarType.BINARY}]

  def train_test_split(self, val_size=0.1, test_size=0.2, **kwargs):
    """
    Splits the Market's data into train, validation, and test sets along the time axis.
    The train set is strictly chronologically less than the val set which is strictly before than the test set.

    Args:
      val_size (float, \\in (0, 1)): proportion of the dataset to include in the validation split.
      test_size (float, \\in (0, 1)): proportion of the dataset to include in the test split.
      **kwargs: optional keyword arguments.
      - `prediction_days` (int &gte 0): number of extra prediction days to include in the train and validation splits.

    Returns: three xr.Datasets for training, validation, and testing.
    """
    num_timesteps = self.xarray_ds['time'].values.shape[0]
    num_train, num_val = int(num_timesteps * (1 - val_size - test_size)), int(num_timesteps * val_size)
    num_train, num_val = max(num_train, 2), max(num_val, 2)

    return (
      self.xarray_ds.isel(time=slice(0, num_train + kwargs.get('prediction_days', 0))),
      self.xarray_ds.isel(time=slice(num_train, num_train + num_val + kwargs.get('prediction_days', 0))),
      self.xarray_ds.isel(time=slice(num_train + num_val, None))
    )
