from abc import ABC, abstractmethod
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt


class Market(ABC):

  def collect_data(self, sep=','):
    pass

  def align_timeseries(self, impute=False):
    common_times = set(self.assets[0].ts_df.time.values)
    for asset in self.assets[1:]:
      common_times = common_times.intersection(asset.ts_df.time.values)

    for asset in self.assets:
      asset.ts_df = asset.ts_df[asset.ts_df.time.isin(common_times)]

  @abstractmethod
  def __init__(self, assets, market_name):
    self.assets = assets
    self.market_name = market_name
    self.levels = {}

  @abstractmethod
  def display_market(self):
    pass


class BaseMarket(Market):
  def __init__(self, assets, market_name=''):
    super().__init__(assets, market_name)

  def display_market(self):
    print(f'Start Market {self.market_name}')
    for asset in self.assets:
      print(asset.ts_df)
    print(f'End Market {self.market_name}')


class XarrayMarket(Market):
  def __init__(self, assets, market_name=''):
    super().__init__(assets, market_name)

    df_combined = pd.concat([asset.ts_df.set_index(['time', 'ID']) for asset in assets], sort=False)
    self.xarray = xr.Dataset.from_dataframe(df_combined).to_dataarray()

  def align_timeseries(self, impute=False):
    if impute:
      # logic
      self.xarray = self.xarray
    else:
      self.xarray = self.xarray.dropna(dim='time')

  def display_market(self):
    # x_ax = np.arange(self.xarray.time.shape[0])
    x_ax = self.xarray.time.values.astype(np.int64)
    y_ax = np.arange(self.xarray.ID.shape[0])
    z_ax = np.arange(self.xarray.variable.shape[0])

    xx, yy, zz = np.meshgrid(x_ax, y_ax, z_ax)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlabel('Time', c='b')
    ax.set_xticks(self.xarray.time)
    # ax.set_xticks(pd.date_range(
    #   start=self.xarray.time.values[0],
    #   end=self.xarray.time.values[-1],
    #   freq="W")
    # )
    ax.set_ylabel('Asset ID', c='b')
    ax.set_yticks(y_ax, [_id[:10] for _id in self.xarray.ID.values], c='m')
    ax.set_zlabel('Measure', c='b')
    ax.set_zticks(z_ax, self.xarray.indexes['variable'].values, c='m')
    ax.set_title(f'3D Visualization of Market {self.market_name}')

    cbar = ax.scatter(
      xx, yy, zz,
      c=self.xarray.to_numpy(), cmap='viridis'
    )

    fig.colorbar(cbar, ax=ax, shrink=0.5, aspect=5, pad=0.1, label='')
    plt.gcf().autofmt_xdate()
    plt.show()