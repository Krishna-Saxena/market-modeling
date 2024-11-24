from abc import ABC, abstractmethod
from enum import Enum, auto

from metrics.ts_utils import *


class VarType(Enum):
  QUANTITATIVE = auto()
  CATEGORICAL = auto()
  BINARY = auto()


class Metric(ABC):
  def __init__(self, var_type: VarType):
    self.var_type = var_type


class CategoricalMetric(Metric):
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
    self.N_REMOVED = 0

  def remove_leading_zeros(self):
    remove_leading_zeros(self)

  def remove_leading_n_values(self, n):
    remove_leading_n_values(self, n)


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
    self.covariates = {key : value for key, value in kwargs.items()}

  def remove_leading_zeros(self):
    super().remove_leading_zeros()
    for key in self.covariates:
      if isinstance(self.covariates[key], Timeseries):
        self.covariates[key].remove_leading_n_values(self.N_REMOVED)