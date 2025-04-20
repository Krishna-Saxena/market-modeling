from abc import ABC, abstractmethod
from typing import Dict

import numpy as np
import xarray as xr

from metrics.Metrics import VarType
from modeling_utils.py_utils import compute_next_ewma, make_ewma_col_name, alpha_to_days, days_to_alpha, \
	make_scale_diff_col_name, make_scale_diff_log_col_name


class DerivedVariable(ABC):
	def __init__(self, op_name, gen_var_types: Dict[str, VarType] = {}, **kwargs):
		"""
		Create a specification for an operation that adds new variable(s) to a xr.Dataset.

		Args:
			op_name: the name of this transform operation.
			gen_var_types: a dict of {generated variable : VarType of the generated variable} for each generated variable.
			**kwargs: each k, v in kwargs becomes a property of this spec.
		"""
		self.op_name = op_name
		self.gen_var_types = gen_var_types
		for key, value in kwargs.items():
			setattr(self, key, value)

	def __str__(self):
		return self.op_name

	@abstractmethod
	def init_var(self, xarr_ds: xr.Dataset, **kwargs) -> xr.Dataset:
		"""
		Logic to initialize variable(s) based on the other variables in a xarray Dataset.

		Args:
			xarr_ds: a xarray Dataset.
			**kwargs: flexible keyword arguments to implement calculations.

		Returns: xarr_ds, with new variable(s) added
		"""
		pass

	@abstractmethod
	def update_var(self, new_xarr_ds, **kwargs) -> xr.Dataset:
		"""
		Given a xarray Dataset with default values in the last time step, update the values of generated variable(s).

		Args:
			new_xarr_ds: a xarray Dataset.
			**kwargs: flexible keyword arguments to implement calculations.

		Returns: new_xarr_ds, with new values in new_xarr_ds.isel(time=-1).sel(columns defined by self.init_var()).
		"""
		pass


class EWMATransform(DerivedVariable):
	def __init__(self, op_name='', gen_var_types: Dict[str, VarType]={}, **kwargs):
		"""
		Create a specification for an exponentially weighted moving average of a variable:
		EWMA_0 = S_0, EWMA_{t+dt} = alpha**dt * EWMA_t + (1 - alpha**dt) * S_{t+dt}.

		Args:
			op_name (str): the name of the EWMA variable created and modified by methods of this spec, default and preferred value is py_utils.make_ewma_col_name(var_to_avg, days) (Note: var_to_avg, days explained in **kwargs).
			**kwargs:
			- alpha (not needed if 'days' passed as kwarg): the decay factor.
			- days (ignored if 'alpha' passed as kwarg): the approximate window size, as used by economists. Equivalent to setting alpha := \frac{2}{days+1}
			- var_to_avg (str): the variable whose EWMA value will be calculated, 'signal' by default.
		"""
		assert 'alpha' in kwargs or 'days' in kwargs, "you must pass one of 'alpha' or 'days' as a kwarg, i.e., EWMATransform(days=7)"
		if 'days' in kwargs:
			kwargs['alpha'] = days_to_alpha(kwargs['days'])
			del kwargs['days']
		if 'var_to_avg' not in kwargs:
			kwargs['var_to_avg'] = 'signal'
		# to-be calculated col's name
		if op_name == '':
			op_name = make_ewma_col_name(kwargs['var_to_avg'], alpha_to_days(kwargs['alpha']))
		super().__init__(op_name, {op_name : VarType.QUANTITATIVE}, **kwargs)

	def init_var(self, xarr_ds: xr.Dataset, **kwargs) -> xr.Dataset:
		base_timeseries = xarr_ds[self.var_to_avg].to_numpy()
		xarr_ds = xarr_ds.assign(**{self.op_name: (['time', 'ID'], base_timeseries[0]*np.ones_like(base_timeseries))})
		dt = (xarr_ds.time - xarr_ds.time.shift({'time': 1})).dt.days.values

		for i in range(1, len(dt)):
			xarr_ds[self.op_name].data[i, :] = compute_next_ewma(
				xarr_ds[self.op_name].data[i-1, :],
				base_timeseries[i],
				dt[i],
				self.alpha
			)

		return xarr_ds

	def update_var(self, new_xarr_ds, **kwargs) -> xr.Dataset:
		"""
		Given a xarray Dataset with default values in the last time step, update the values of this variable.

		Args:
			new_xarr_ds: a xarray Dataset.
			**kwargs:
			- dt (optional) - delta time to between the -2nd (last known) and -1st (first generated) timestep.

		Returns: new_xarr_ds, with a new value in new_xarr_ds.isel(time=-1)[self.op_name].
		"""
		dt = kwargs.get('dt', (new_xarr_ds.time - new_xarr_ds.time.shift({'time': 1})).dt.days.values[-1])
		new_xarr_ds[self.op_name].data[-1, :] = compute_next_ewma(
			new_xarr_ds[self.op_name].data[-2, :],
			new_xarr_ds[self.var_to_avg].data[-1, :],
			dt,
			self.alpha
		)

		return new_xarr_ds


class ScaledDeltaTransform(DerivedVariable):
	def __init__(self, op_name='', gen_var_types: Dict[str, VarType]={}, **kwargs):
		"""
		Create a specification for a scaled difference transform of a variable:
		`op_name`_0 = 0, `op_name`_{t+dt} = (var_to_transf_{t+dt} - var_to_transf_t)/dt.

		Args:
			op_name (str): the name of the delta variable created and modified by methods of this spec, default and preferred value is py_utils.make_scale_diff_col_name(var_to_avg, days).
			gen_var_types: a dict of {generated variable : VarType of the generated variable} for each generated variable.
			**kwargs:
			- var_to_transf (str): the variable to transform, 'signal' by default.
		"""
		if 'var_to_transf' not in kwargs:
			kwargs['var_to_transf'] = 'signal'
		# to-be calculated col's name
		if op_name == '':
			op_name = make_scale_diff_col_name(kwargs['var_to_transf'])
		super().__init__(op_name, {op_name : VarType.QUANTITATIVE}, **kwargs)

	def init_var(self, xarr_ds: xr.Dataset, **kwargs) -> xr.Dataset:
		base_timeseries = xarr_ds[self.var_to_transf].to_numpy()
		xarr_ds = xarr_ds.assign(**{self.op_name: (['time', 'ID'], np.zeros_like(base_timeseries))})
		dt = (xarr_ds.time - xarr_ds.time.shift({'time': 1})).dt.days.values

		xarr_ds[self.op_name].data[1:, :] = (base_timeseries[1:, :] - base_timeseries[:-1, :]) / dt[1:, None]

		return xarr_ds

	def update_var(self, new_xarr_ds, **kwargs) -> xr.Dataset:
		"""
		Given a xarray Dataset with default values in the last time step, update the values of this variable.

		Args:
			new_xarr_ds: a xarray Dataset.
			**kwargs:
			- dt (optional) - delta time to between the -2nd (last known) and -1st (first generated) timestep.

		Returns: xarr_ds, with a new value in new_xarr_ds.isel(time=-1)[self.op_name].
		"""
		dt = kwargs.get('dt', (new_xarr_ds.time - new_xarr_ds.time.shift({'time': 1})).dt.days.values[-1])
		new_xarr_ds[self.op_name].data[-1, :] = (new_xarr_ds[self.var_to_transf].data[-1, :] - new_xarr_ds[self.var_to_transf].data[-2, :])/dt
		return new_xarr_ds


class ScaledDeltaLogTransform(DerivedVariable):
	def __init__(self, op_name='', gen_var_types: Dict[str, VarType]={}, **kwargs):
		"""
		Create a specification for a scaled log-difference transform of a variable:
		`op_name`_0 = 0, `op_name`_{t+dt} = (ln(var_to_transf_{t+dt}) - ln(var_to_transf_{t}))/dt

		Args:
			op_name: the name of the delta log variable created and modified by methods of this spec.
			gen_var_types: a dict of {generated variable : VarType of the generated variable} for each generated variable.
			**kwargs:
			- var_to_transf (str): the variable to transform, 'signal' by default.
		"""
		if 'var_to_transf' not in kwargs:
			kwargs['var_to_transf'] = 'signal'
		# to-be calculated col's name
		if op_name == '':
			op_name = make_scale_diff_log_col_name(kwargs['var_to_transf'])
		super().__init__(op_name, {op_name : VarType.QUANTITATIVE}, **kwargs)

	def init_var(self, xarr_ds: xr.Dataset, **kwargs) -> xr.Dataset:
		log_base_timeseries = np.log(xarr_ds[self.var_to_transf].to_numpy())
		xarr_ds = xarr_ds.assign(**{self.op_name: (['time', 'ID'], np.zeros_like(log_base_timeseries))})
		dt = (xarr_ds.time - xarr_ds.time.shift({'time': 1})).dt.days.values

		xarr_ds[self.op_name].data[1:, :] = (log_base_timeseries[1:, :] - log_base_timeseries[:-1, :]) / dt[1:, None]

		return xarr_ds

	def update_var(self, new_xarr_ds, **kwargs) -> xr.Dataset:
		"""
		Given a xarray Dataset with default values in the last time step, update the values of this variable.

		Args:
			new_xarr_ds: a xarray Dataset.
			**kwargs:
			- dt (optional) - delta time to between the -2nd (last known) and -1st (first generated) timestep.

		Returns: xarr_ds, with a new value in new_xarr_ds.isel(time=-1)[self.op_name].
		"""
		dt = kwargs.get('dt', (new_xarr_ds.time - new_xarr_ds.time.shift({'time': 1})).dt.days.values[-1])
		new_xarr_ds[self.op_name].data[-1, :] = (np.log(new_xarr_ds[self.var_to_transf].data[-1, :]) - np.log(new_xarr_ds[self.var_to_transf].data[-2, :]))/dt
		return new_xarr_ds