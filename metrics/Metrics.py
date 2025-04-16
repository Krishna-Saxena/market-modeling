from abc import ABC
from enum import Enum, auto

from metrics.ts_utils import *

import pandas as pd


class VarType(Enum):
  QUANTITATIVE = auto()
  BINARY = auto()
  CATEGORICAL = auto()
  MULTI_CATEGORICAL = auto()


class Metric(ABC):
  def __init__(self, var_type: VarType, value):
    self.var_type = var_type
    self.value = value


class StaticMetric(Metric):
  def __init__(self, value, var_type: VarType):
    super().__init__(var_type, value)


class TimeseriesMetric(Metric):
  def __init__(self, time, signal, signal_var_type: VarType):
    """
    Args:
      time: a 1D numpy array/torch tensor of days (since first measurement)
      signal: a 1D numpy array/torch tensor of measurements synced to `time`
      signal_var_type: a Metric.VarType enum describing the type of `signal`
    """
    super().__init__(signal_var_type, None)
    self.value = pd.DataFrame({'time': time, 'signal': signal})
    self.N_REMOVED = 0
    remove_leading_zeros(self)

  def remove_leading_zeros(self):
    """
    Removes all initial observations from `self.value` where `signal == 0.0` (and their associated timestamps), in place.
    Saves the number of elements removed into `self.N_REMOVED`

    """
    remove_leading_zeros(self)

  def remove_leading_n_values(self, n: int):
    """
    Remove the first n entries from `timeseries`'s time and signal vectors, in place.
    Sets self.N_REMOVED to `n`

    """
    remove_leading_n_values(self, n)