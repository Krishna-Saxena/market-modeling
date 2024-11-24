def remove_leading_zeros(signal, time):
  """
  Remove all leading zeros from signal and their associated times

  Args:
    signal:
    time: a 1D numpy array/torch tensor of days (since a student started practicing)

  Returns:

  """
  first_non_zero = 0
  while signal[first_non_zero] == 0:
    first_non_zero += 1
  time = time[first_non_zero:]
  signal = signal[first_non_zero:]
  return signal, time, first_non_zero