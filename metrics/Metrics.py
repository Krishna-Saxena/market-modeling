from abc import ABC, abstractmethod
from enum import Enum, auto


class VarType(Enum):
  QUANTITATIVE = auto()
  CATEGORICAL = auto()
  BINARY = auto()


class Metric(ABC):
  def __init__(self, var_type: VarType):
    self.var_type = var_type


class CategoricalMetric(ABC, Metric):
  levels = {}

  @property
  @abstractmethod
  def var_type(self):
    pass


class Timeseries(Metric):
  @abstractmethod
  def __init__(self, time, signal, signal_var_type: VarType, **kwargs):
    """
    Args:
      time: a 1D numpy array/torch tensor of days (since first measurement)
      signal: a 1D numpy array/torch tensor of measurements synced to `time`
      signal_var_type: a Metric.VarType enum describing the type of `signal`
      **kwargs: additional 1D numpy array/torch tensors of measurements synced to `time`
    """
    super().__init__(signal_var_type)
    self.time = time
    self.signal = signal
    self.var_type = signal_var_type


class BaseTimeseries(Timeseries):
  def __init__(self, time, signal, signal_var_type: VarType, **kwargs):
    """
    Args:
      time: a 1D numpy array/torch tensor of days (since first measurement)
      signal: a 1D numpy array/torch tensor of measurements synced to `time`
      signal_var_type: a Metric.VarType enum describing the type of `signal`
      **kwargs: NOT USED in BaseTimeseries
    """
    super().__init__(time, signal, signal_var_type)


class CovariateTimeseries(Timeseries):
  def __init__(self, time, signal, signal_var_type: VarType, **kwargs):
    super().__init__(time, signal, signal_var_type)
    for key, value in kwargs.items():
      setattr(self, key, value)
