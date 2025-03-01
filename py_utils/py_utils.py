def last_index_lt_1D(a, value):
  """

  Args:
    a: a 1D numpy array
    value: a value satisfying type(value) == a.dtype

  Returns: the last index i : a[i] < value

  """
  # https://stackoverflow.com/a/42945171
  idx = (a < value)[::-1].argmax()
  return a.shape[0] - idx - 1

def fmt_name(first, last):
  first = '' if not first else first
  last = '' if not last else last
  return f'{first} {last}'