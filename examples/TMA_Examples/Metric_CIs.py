from datetime import datetime
from collections import defaultdict
import warnings

from examples.TMA_Examples import get_markets, get_metrics_df_from_hist
from markets.DerivedVariables import DerivedVariable
from metrics.EvalMetrics import evaluate_no_cov_model, evaluate_cov_model

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from metrics.Metrics import VarType
from modeling_utils.py_utils import make_ewma_col_name, make_scale_diff_log_col_name, days_to_alpha

DEBUG = False
INTERNAL = False
DATA_DIR = '../../data/TMA_Datasets'
TS_DIR = 'KA_timeseries'

chosen_covars=[
	'duration_mins', 'at_home_mins', 'total_activities',
	'completion_percentage', 'restarts_per_activity',
	'attendance', 'attendance_60',
	'grade_2', 'grade_3', 'grade_4', 'grade_5'
]

class MasterTransform(DerivedVariable):
	def __init__(self, op_name, **kwargs):
		gen_var_types = {
			'problem_percentage': VarType.QUANTITATIVE,
			'completion_percentage': VarType.QUANTITATIVE,
			'restarts_per_activity': VarType.QUANTITATIVE,
			'cumsum_present': VarType.QUANTITATIVE,
			'cumsum_absent': VarType.QUANTITATIVE,
			'cumsum_present_60': VarType.QUANTITATIVE,
			'cumsum_absent_60': VarType.QUANTITATIVE,
			'attendance': VarType.QUANTITATIVE,
			'attendance_60': VarType.QUANTITATIVE
		}
		super().__init__(op_name, gen_var_types, **kwargs)

	def init_var(self, xr_ds, **kwargs):
		"""
		Logic to initialize variable(s) based on the other variables in a xarray Dataset.

		Args:
			xr_ds: a xarray Dataset.
			**kwargs: flexible keyword arguments to implement calculations.

		Returns: xarr_ds, with new variable(s) added
		"""
		div0_eps = 1e-3

		# Performance covars
		xr_ds['problem_percentage'] = xr_ds['correct_count'] / (xr_ds['problem_count'] + div0_eps)
		xr_ds['completion_percentage'] = xr_ds['complete_activities'] / (xr_ds['complete_activities'] + xr_ds['incomplete_activities'] + div0_eps)
		xr_ds['restarts_per_activity'] = xr_ds['num_restarts'] / (xr_ds['total_activities'] + div0_eps)

		# Attendance covars
		# 	create new variables that calculate total number of sessions students are present and absent
		xr_ds[['cumsum_present', 'cumsum_absent']] = xr_ds[['present', 'absent']].cumsum('time')
		# 	the same, but only considering the last ~2 months students have practiced (2 sessions/week * 8weeks/month = 16 sessions + 4 bonus for at-home)
		xr_ds[['cumsum_present_60', 'cumsum_absent_60']] = xr_ds[['present', 'absent']].rolling(dim={'time':20}, min_periods=1).sum()
		# 	calculate attendance rates
		xr_ds['attendance'] = xr_ds['cumsum_present'] / (xr_ds['cumsum_present'] + xr_ds['cumsum_absent'] + div0_eps)
		xr_ds['attendance_60'] = xr_ds['cumsum_present_60'] / (xr_ds['cumsum_present_60'] + xr_ds['cumsum_absent_60'] + div0_eps)

		return xr_ds

	def update_var(self, new_xr_ds, **kwargs):
		"""
		Given a xarray Dataset with default values in the last time step, update the values of generated variable(s).

		Args:
			new_xr_ds: a xarray Dataset.
			**kwargs: flexible keyword arguments to implement calculations.
			- dt (int, optional) - delta time to between the -2nd (last known) and -1st (first generated) timestep.

		Returns: new_xarr_ds, with new values in new_xarr_ds.isel(time=-1).sel(columns defined by self.init_var()).
		"""

		return new_xr_ds


def get_one_eval(model_name):
	# Step 1: Instantiate the correct model
	if model_name == 'AI':
		from models.arith_growth_models.Models import IndependentModel
		model = IndependentModel()
	elif model_name == 'AD':
		from models.arith_growth_models.Models import DependentModel
		model = DependentModel()
	elif model_name == 'ADC':
		from models.arith_growth_models.Models import DependentCovariateModel
		model = DependentCovariateModel(covar_vars=chosen_covars)
	elif model_name == 'GI':
		from models.geom_growth_models.Models import IndependentModel
		model = IndependentModel()
	elif model_name == 'GD':
		from models.geom_growth_models.Models import DependentModel
		model = DependentModel()
	elif model_name == 'GDC':
		from models.geom_growth_models.Models import DependentCovariateModel
		model = DependentCovariateModel(covar_vars=chosen_covars)
	else:
		raise ValueError('Invalid model_name')

	# Step 2: Create train, val, test splits
	markets = get_markets(DATA_DIR, TS_DIR, (model_name[0] == 'G'), DEBUG, INTERNAL)

	for market in markets:
		# One/Multi hot encode hierarchical info
		# market.encode_indicators(True, ',')
		market.register_derived_var(MasterTransform('master'))
		market.align_timeseries(False)

	data_splits = [market.train_test_split(0.2, 0.1) for market in markets]

	# Step 3: Evaluate
	if model_name == 'ADC':
		theta, _ = model.fit_global_params([split[0] for split in data_splits], save_hist=False)
		return evaluate_cov_model(model, theta, markets, data_splits)
	elif model_name == 'GDC':
		theta, _ = model.fit_global_params(
			[split[0] for split in data_splits],
			[split[1] for split in data_splits],
			train_epochs=30,
			steps_per_batch=200,
			val_freq=2,
			var_types=markets[0].var_types,
			der_var_specs=markets[0].derived_variables,
			lr_decay=0.1**0.5,
			l1_penalty=0.,
			save_hist=False
		)
		return evaluate_cov_model(model, theta, markets, data_splits)
	else:
		return evaluate_no_cov_model(model, markets, data_splits)

def create_CIs(model_name, n_iters=10):
	metric_history = defaultdict(list)

	for i in range(n_iters):
		eval_metrics = get_one_eval(model_name)
		for metric_name, metric_value in eval_metrics.items():
			metric_history[metric_name].append(metric_value)

	metrics_df = get_metrics_df_from_hist(metric_history)
	metrics_df['N'] = n_iters
	return metrics_df