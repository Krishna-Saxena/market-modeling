from abc import ABC, abstractmethod

import pandas as pd

from metrics.Metrics import Metric, VarType, StaticMetric, TimeseriesMetric
from modeling_utils.py_utils import ffill_df, bfill_df


class Asset(ABC):
  @abstractmethod
  def __init__(self, name, asset_id, metrics: {str: Metric}):
    """
    Create an Asset with the given metrics synced to a series of times

    Args:
      name: the name of the asset
      asset_id: a unique id for the asset within a market
      metrics: a dict of {name_var : val_var} where val_var can be Metrics.StaticMetric or Metrics.TimeseriesMetric
        and all timeseries metrics are pre-synced
    """
    self.name = name
    self.asset_id = asset_id
    if 'signal' not in metrics:
      raise TypeError('metrics must have a Metrics.TimeseriesMetric value under key signal')
    if not isinstance(metrics['signal'], TimeseriesMetric):
      raise TypeError('signal Metric must be a TimeseriesMetric')
    if metrics['signal'].var_type != VarType.QUANTITATIVE:
      raise TypeError('signal must be a Metrics.TimeseriesMetric of type VarType.QUANTITATIVE')
    self.ts_df = None
    self.var_types = {}

  def ffill(self, signal_ffill_fn, **kwargs):
    """
    Forward fill an Asset's time series metrics.

    Args:
      signal_ffill_fn: a function that takes a timeseries DataFrame and returns it with the signal column filled.
      **kwargs: kwargs passed to signal_ffill_fn
      - num_sims: number of simulations for each NaN value, 100 by default.
      - col_to_fill: a column in `ts_df` whose NaN values will be filled, 'signal' by default.

    Returns: None. Modifies `self.ts_df`.
    """
    self.ts_df = ffill_df(self.ts_df, ('ID', 'time', 'signal'))
    self.ts_df = signal_ffill_fn(self.ts_df, **kwargs)

  def bfill(self, signal_bfill_fn, **kwargs):
    """
    Backward fill an Asset's time series metrics.

    Args:
      signal_bfill_fn: a function that takes a timeseries DataFrame and returns it with the signal column filled.
      **kwargs: kwargs passed to signal_bfill_fn
      - num_sims: number of simulations for each NaN value, 100 by default.
      - col_to_fill: a column in `ts_df` whose NaN values will be filled, 'signal' by default.

    Returns: None. Modifies `self.ts_df`.
    """
    self.ts_df = bfill_df(self.ts_df, ('ID', 'time', 'signal'))
    self.ts_df = signal_bfill_fn(self.ts_df, **kwargs)


class BaseAsset(Asset):
  def __init__(self, name, asset_id, metrics: {str: Metric}):
    """
    Same behavior as Asset but requires that metrics['signal'] is a Metrics.Timeseries
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


class CovariateAsset(BaseAsset):
  def __init__(self, name, asset_id, metrics: {str: Metric}):
    super().__init__(name, asset_id, metrics)
    for metric_name, metric in metrics.items():
      self.var_types[metric_name] = metric.var_type
      if isinstance(metric, TimeseriesMetric):
        self.ts_df[metric_name] = metric.value['signal']
      else:
        self.ts_df[metric_name] = metric.value


class HeirarchicalAsset(CovariateAsset):
  """
  Same behavior as Asset but requires that metrics['signal'] is a Metrics.Timeseries,
    metrics['sector_id'] returns a Metrics.StaticMetric with VarType=CATEGORICAL,
    metrics['industry_id'] returns a Metrics.StaticMetric with VarType=CATEGORICAL
  """

  def __init__(self, name, asset_id, metrics: {str: Metric}):
    assert isinstance(metrics['sector_id'], StaticMetric)
    assert metrics['sector_id'].var_type == VarType.CATEGORICAL
    assert isinstance(metrics['industry_id'], StaticMetric)
    assert metrics['industry_id'].var_type == VarType.CATEGORICAL
    super().__init__(name, asset_id, metrics)
