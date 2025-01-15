from abc import ABC, abstractmethod

import pandas as pd

from metrics.Metrics import Metric, VarType, StaticMetric, TimeseriesMetric


class Asset(ABC):

  @abstractmethod
  def __init__(self, name, asset_id, metrics: {str: Metric}):
    """
    Create an Asset with the given metrics synced to a series of times

    Args:
      name: the name of the asset
      asset_id: a unique id for the asset within a market
      metrics: a dict of {name_var : val_var} where val_var can be Metrics.StaticMetric or Metrics.Timeseries
        and all timeseries metrics are pre-synced
    """
    self.name = name
    self.asset_id = asset_id
    if not isinstance(metrics['signal'], TimeseriesMetric):
      raise TypeError('signal Metric must be a TimeseriesMetric')
    if metrics['signal'].var_type != VarType.QUANTITATIVE:
      raise TypeError('signal Metric must be a Timeseries of type VarType.QUANTITATIVE')
    self.ts_df = None
    self.var_types = {}


class BaseAsset(Asset):
  def __init__(self, name, asset_id, metrics: {str: Metric}):
    """
    Same behavior as Asset but requires that metrics['signal'] returns a Metrics.Timeseries
    """
    super().__init__(name, asset_id, metrics)
    self.ts_df = pd.DataFrame(
      {
        'ID' : [asset_id]*len(metrics['signal'].value),
        'time' : metrics['signal'].value.time,
        'signal' : metrics['signal'].value.signal
      }
    )
    self.var_types = {'signal': metrics['signal'].var_type}


class CovariateAsset(Asset):
  def __init__(self, name, asset_id, growth_timeseries, **kwargs):
    # TODO: save the data in kwargs in ts_df (instead of as attributes)
    super().__init__(name, asset_id, growth_timeseries)
    for key, value in kwargs.items():
      setattr(self, key, value)


class HeirarchicalAsset(Asset):
  """
  Same behavior as Asset but requires that metrics['signal'] returns a Metrics.Timeseries,
    metrics['sector_id'] returns a Metrics.StaticMetric with VarType=CATEGORICAL,
    metrics['industry_id'] returns a Metrics.StaticMetric with VarType=CATEGORICAL
  """

  def __init__(self, name, asset_id, **kwargs):
    # TODO: save the data in kwargs in ts_df (instead of as attributes)

    super().__init__(name, asset_id, kwargs)
    assert isinstance(kwargs['sector_id'], StaticMetric)
    assert kwargs['sector_id'].var_type == VarType.CATEGORICAL
    assert isinstance(kwargs['industry_id'], StaticMetric)
    assert kwargs['industry_id'].var_type == VarType.CATEGORICAL

    for key, value in kwargs.items():
      setattr(self, key, value)