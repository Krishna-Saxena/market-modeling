from abc import ABC, abstractmethod

import numpy as np
import xarray as xr

from modeling_utils.py_utils import compute_next_ewma


class DerivedVariable(ABC):
	def __init__(self, var_name, **kwargs):
		"""
		Create a specification for a new variable in a xarray dataset.

		Args:
			var_name: the name of the variable created and modified by methods of this spec.
			**kwargs: each k, v in kwargs becomes a property of this spec.
		"""
		self.var_name = var_name
		for key, value in kwargs.items():
			setattr(self, key, value)

	def __str__(self):
		return self.var_name

	@abstractmethod
	def init_var(self, xarr_ds: xr.Dataset, **kwargs):
		"""
		Logic to initialize a variable based on the other variables in a xarray Dataset.

		Args:
			xarr_ds: a xarray Dataset.
			**kwargs: flexible keyword arguments to implement calculations.

		Returns: xarr_ds, with a new variable named self.var_name.
		"""
		pass

	@abstractmethod
	def update_var(self, new_xarr_ds, **kwargs):
		"""
		Given a xarray Dataset with default values in the last time step, update the values of this variable.

		Args:
			new_xarr_ds: a xarray Dataset.
			**kwargs: flexible keyword arguments to implement calculations.

		Returns: new_xarr_ds, with a new value in new_xarr_ds.isel(time=-1)[self.var_name].
		"""
		pass


class EWMATransform(DerivedVariable):
	def __init__(self, var_name, **kwargs):
		"""
		Create a specification for an exponentially weighted moving average of a variable.

		Args:
			var_name: the name of the EWMA variable created and modified by methods of this spec.
			**kwargs:
			- alpha (not needed if 'days' passed as kwarg): the decay factor.
			- days (ignored if 'alpha' passed as kwarg): the approximate window size, as used by economists. Equivalent to setting alpha := \frac{2}{days+1}
			- var_to_avg (str): the variable whose EWMA value will be calculated, 'signal' by default.
		"""
		assert 'alpha' in kwargs or 'days' in kwargs, "you must pass one of 'alpha' or 'days' as a kwarg, i.e., EWMATransform('7_day_transform', days=7)"
		if 'days' in kwargs:
			kwargs['alpha'] = 2./(kwargs['days'] + 1)
			del kwargs['days']
		if 'var_to_avg' not in kwargs:
			kwargs['var_to_avg'] = 'signal'
		super().__init__(var_name, **kwargs)

	def __str__(self):
		return '{} EWMA decayed by {.4f} ~ {} days'.format(self.var_to_avg, self.alpha, int(2/self.alpha - 1))

	def init_var(self, xarr_ds: xr.Dataset, **kwargs):
		base_timeseries = xarr_ds[self.var_to_avg].to_numpy()
		xarr_ds = xarr_ds.assign(**{self.var_name: (['time', 'ID'], base_timeseries[0]*np.ones_like(base_timeseries))})
		dt = (xarr_ds.time - xarr_ds.time.shift({'time': 1})).dt.days.values

		for i in range(1, len(dt)):
			xarr_ds[self.var_name].data[i, :] = compute_next_ewma(
				xarr_ds[self.var_name].data[i-1, :],
				base_timeseries[i],
				dt[i],
				self.alpha
			)

		return xarr_ds

	def update_var(self, new_xarr_ds, **kwargs):
		"""
		Given a xarray Dataset with default values in the last time step, update the values of this variable.

		Args:
			new_xarr_ds: a xarray Dataset.
			**kwargs:
			- dt (optional) - delta time to between the -2nd (last known) and -1st (first generated) timestep.

		Returns: new_xarr_ds, with a new value in new_xarr_ds.isel(time=-1)[self.var_name].
		"""
		dt = kwargs.get('dt', (new_xarr_ds.time - new_xarr_ds.time.shift({'time': 1})).dt.days.values[-1])
		new_xarr_ds[self.var_name].data[-1, :] = compute_next_ewma(
			new_xarr_ds[self.var_name].data[-2, :],
			new_xarr_ds[self.var_to_avg].data[-1, :],
			dt,
			self.alpha
		)

		return new_xarr_ds


class ScaledDeltaTransform(DerivedVariable):
	def __init__(self, var_name, **kwargs):
		"""
		Create a specification for a scaled difference transform of a variable:
		var_name_{t+dt} = (var_to_avg_{t+dt} - var_to_avg_t)/dt

		Args:
			var_name: the name of the delta variable created and modified by methods of this spec.
			**kwargs:
			- var_to_transf (str): the variable to transform, 'signal' by default.
		"""
		if 'var_to_transf' not in kwargs:
			kwargs['var_to_transf'] = 'signal'
		super().__init__(var_name, **kwargs)

	def __str__(self):
		return '{} delta'.format(self.var_to_transf)

	def init_var(self, xarr_ds: xr.Dataset, **kwargs):
		base_timeseries = xarr_ds[self.var_to_transf].to_numpy()
		xarr_ds = xarr_ds.assign(**{self.var_name: (['time', 'ID'], np.zeros_like(base_timeseries))})
		dt = (xarr_ds.time - xarr_ds.time.shift({'time': 1})).dt.days.values

		xarr_ds[self.var_name].data[1:, :] = (base_timeseries[1:, :] - base_timeseries[:-1, :]) / dt[1:, None]

		return xarr_ds

	def update_var(self, new_xarr_ds, **kwargs):
		"""
		Given a xarray Dataset with default values in the last time step, update the values of this variable.

		Args:
			new_xarr_ds: a xarray Dataset.
			**kwargs:
			- dt (optional) - delta time to between the -2nd (last known) and -1st (first generated) timestep.

		Returns: xarr_ds, with a new value in new_xarr_ds.isel(time=-1)[self.var_name].
		"""
		dt = kwargs.get('dt', (new_xarr_ds.time - new_xarr_ds.time.shift({'time': 1})).dt.days.values[-1])
		new_xarr_ds[self.var_name].data[-1, :] = (new_xarr_ds[self.var_to_transf].data[-1, :] - new_xarr_ds[self.var_to_transf].data[-2, :])/dt
		return new_xarr_ds


class ScaledDeltaLogTransform(DerivedVariable):
	def __init__(self, var_name, **kwargs):
		"""
		Create a specification for a scaled log-difference transform of a variable:
		var_name_{t+dt} = (ln(var_to_avg_{t+dt}) - ln(var_to_avg_t))/dt

		Args:
			var_name: the name of the delta log variable created and modified by methods of this spec.
			**kwargs:
			- var_to_transf (str): the variable to transform, 'signal' by default.
		"""
		if 'var_to_transf' not in kwargs:
			kwargs['var_to_transf'] = 'signal'
		super().__init__(var_name, **kwargs)

	def __str__(self):
		return '{} delta log'.format(self.var_to_transf)

	def init_var(self, xarr_ds: xr.Dataset, **kwargs):
		log_base_timeseries = np.log(xarr_ds[self.var_to_transf].to_numpy())
		xarr_ds = xarr_ds.assign(**{self.var_name: (['time', 'ID'], np.zeros_like(log_base_timeseries))})
		dt = (xarr_ds.time - xarr_ds.time.shift({'time': 1})).dt.days.values

		xarr_ds[self.var_name].data[1:, :] = (log_base_timeseries[1:, :] - log_base_timeseries[:-1, :]) / dt[1:, None]

		return xarr_ds

	def update_var(self, new_xarr_ds, **kwargs):
		"""
		Given a xarray Dataset with default values in the last time step, update the values of this variable.

		Args:
			new_xarr_ds: a xarray Dataset.
			**kwargs:
			- dt (optional) - delta time to between the -2nd (last known) and -1st (first generated) timestep.

		Returns: xarr_ds, with a new value in new_xarr_ds.isel(time=-1)[self.var_name].
		"""
		dt = kwargs.get('dt', (new_xarr_ds.time - new_xarr_ds.time.shift({'time': 1})).dt.days.values[-1])
		new_xarr_ds[self.var_name].data[-1, :] = (np.log(new_xarr_ds[self.var_to_transf].data[-1, :]) - np.log(new_xarr_ds[self.var_to_transf].data[-2, :]))/dt
		return new_xarr_ds