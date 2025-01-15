from abc import ABC, abstractmethod
from collections import defaultdict

from metrics.Metrics import VarType


class Market(ABC):

  def collect_data(self, sep=','):
    pass

  def align_timeseries(self):
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


class BaseMarket(Market):
  def __init__(self, assets, market_name=''):
    super().__init__(assets, market_name)
