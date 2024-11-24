from abc import ABC, abstractmethod


class Market(ABC):

  @abstractmethod
  def collect_data(self):
    pass

  @abstractmethod
  def __init__(self, assets):
    self.assets = assets


class BaseMarket(Market):
  def __init__(self, assets, market_name=''):
    self.assets = assets
    self.market_name = market_name

  def collect_data(self):
    raise NotImplementedError('base market')

  def initialize(self):
    raise NotImplementedError('base market')
