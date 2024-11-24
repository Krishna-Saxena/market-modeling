def remove_leading_zeros(timeseries):
  """
  Removes all leading zeros from timeseries.signal and their associated times, in place.
  Saves the number of elements removed into timeseries.N_REMOVED

  Args:
    timeseries: a Metrics.Timeseries object

  """
  first_non_zero = 0
  while timeseries.signal[first_non_zero] == 0:
    first_non_zero += 1
  timeseries.time = timeseries.time[first_non_zero:]
  timeseries.signal = timeseries.signal[first_non_zero:]
  timeseries.N_REMOVED = first_non_zero


def remove_leading_n_values(timeseries, n: int):
  """
  Remove the first n entries from `timeseries`'s time and signal vectors, in place.

  Args:
    timeseries: Metrics.Timeseries object
    n: the number of entries to remove

  """
  timeseries.time = timeseries.time[n:]
  timeseries.signal = timeseries.signal[n:]
  timeseries.N_REMOVED = n
