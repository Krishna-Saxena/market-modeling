from abc import ABC, abstractmethod

class Model(ABC):
  def __init__(self, market):
    self.market = market
    self.N_ASSETS = len(self.market.assets)

  @abstractmethod
  def estimate_parameters(self, **kwargs):
    pass

  @abstractmethod
  def simulate(self, S_0, dates, num_sims, **kwargs):
    """
    simulate future market behavior

    Args:
      S_0: the market state at the start of the simulation, shaped (self.N_ASSETS,)
      dates: the dates to simulate, with dates[0] being the simulation's start time
      num_sims: number of simulations
      kwargs: optional keyword arguments
        add_BM: bool indicating whether to add Brownian Motion (randomness) to simulation

    Returns: a 3D numpy array [num_sims x self.N_ASSETS x len(dates)]
    """
    pass