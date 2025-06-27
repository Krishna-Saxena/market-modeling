from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from Metrics import TimeseriesMetric


def remove_leading_zeros_fn(timeseries: 'TimeseriesMetric'):
	"""
	Removes all leading zeros from timeseries.signal and their associated times, in place.
	Saves the number of elements removed into timeseries.N_REMOVED

	Args:
		timeseries: a Metrics.Timeseries object

	"""
	first_non_zero = 0
	while timeseries.value.signal.iloc[first_non_zero] == 0:
		first_non_zero += 1
	timeseries.value.time = timeseries.value.time[first_non_zero:]
	timeseries.value.signal = timeseries.value.signal[first_non_zero:]
	timeseries.N_REMOVED = first_non_zero


def remove_leading_n_values_fn(timeseries: 'TimeseriesMetric', n: int):
	"""
	Remove the first n entries from `timeseries`'s time and signal vectors, in place.

	Args:
		timeseries: Metrics.Timeseries object
		n: the number of entries to remove

	"""
	timeseries.value = timeseries.value.iloc[n:, :]
	timeseries.N_REMOVED = n
