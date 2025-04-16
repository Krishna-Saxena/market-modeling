def get_zero_grad_hook(mask):
  def hook(grad):
    return grad * mask
  return hook