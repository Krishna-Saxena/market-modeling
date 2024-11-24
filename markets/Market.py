from abc import ABC, abstractmethod


class Market(ABC):
  @abstractmethod
  def collect_data(self):
    pass

  @abstractmethod
  def align_timeseries(self):
    pass

  @abstractmethod
  def __init__(self, assets, market_name):
    self.assets = assets
    self.market_name = market_name
    self.levels = {}


class BaseMarket(Market):
  def __init__(self, assets, market_name=''):
    super().__init__(assets, market_name)

  def collect_data(self):
    raise NotImplementedError('base market')
