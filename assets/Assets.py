from abc import ABC, abstractmethod


class Asset(ABC):

  @abstractmethod
  def __init__(self, name, asset_id, growth_timeseries):
    self.name = name
    self.asset_id = asset_id
    self.growth_timeseries = growth_timeseries

  @abstractmethod
  def collect_data(self):
    pass


class BaseAsset(Asset):
  def __init__(self, name, asset_id, growth_timeseries):
    super().__init__(name, asset_id, growth_timeseries)

  def collect_data(self):
    pass


class CovariateAsset(Asset):
  def __init__(self, name, asset_id, growth_timeseries, **kwargs):
    super().__init__(name, asset_id, growth_timeseries)
    for key, value in kwargs.items():
      setattr(self, key, value)

  def collect_data(self):
    raise NotImplementedError('CovariateAsset.collect_data()')


class HeirarchicalAsset(Asset):
  def __init__(self, name, asset_id, sector_id, industry_id, growth_timeseries):
    super().__init__(name, asset_id, growth_timeseries)
    self.sector_id, self.industry_id = sector_id, industry_id

  def collect_data(self):
    raise NotImplementedError('HeirarchicalAsset.collect_data()')


class HeirarchicalCovariateAsset(HeirarchicalAsset, CovariateAsset):
  def __init__(self, name, asset_id, sector_id, industry_id, growth_timeseries, **kwargs):
    super().__init__(name, asset_id, sector_id, industry_id, growth_timeseries)
    for key, value in kwargs.items():
      setattr(self, key, value)

  def collect_data(self):
    raise NotImplementedError('HeirarchicalCovariateAsset.collect_data()')
