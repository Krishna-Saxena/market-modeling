import numpy as np

def mae(y_true, y_pred):
  return np.mean(np.abs(y_pred - y_true) / y_true)