import numpy as np

from modeling_utils.py_utils import add_random_noise_to_Cov_mat

def get_mle_dep_SRW_params(dS, dt):
	"""
	Return the MLE parameters (mu, Sigma) for a Simple Random Walk model:
	vec{S}_{t+dt} = vec{S_t} + dt*vec{mu} + N(0, dt*Sigma)

	Args:
		dS: observations of vec{S}_{t+dt} - vec{S_t}. Shaped (n_assets, n_timestamps)
		dt: observations of dt. Shaped (n_timestamps)

	Returns: mu, Sigma, where mu, Sigma are shaped (n_assets, ) and (n_assets, n_assets).
	"""
	delta_S = np.sum(dS, axis=1)
	delta_t = np.sum(dt)

	mu_tilde = delta_S / delta_t
	scaled_mu = np.einsum('a,t->at', mu_tilde, dt)
	err = dS - scaled_mu
	sigma_tilde = np.einsum('ij,kj->ik', err/ dt[None, :], err)/dt.shape[0]
	return mu_tilde, sigma_tilde

def get_mle_dep_Cov_SRW_params(dS, dt, Covs):
	"""
	Return the MLE parameters (mu, Sigma) for a Simple Random Walk model:
	vec{S}_{t+dt} = vec{S_t} + dt*(vec{mu} + Covs_t@theta) + N(0, dt*Sigma)

	Args:
		dS: observations of vec{S}_{t+dt} - vec{S_t}. Shaped (n_assets, n_timestamps).
		dt: observations of dt. Shaped (n_timestamps, ).

	Returns: (mu, Sigma, theta), where mu, Sigma are shaped (n_assets, ) and (n_assets, n_assets).
	"""
	n_a, n_t, n_c = Covs.shape

	# s = n_a+n_c
	eye_Covs_aug = np.concatenate((np.eye(n_a)[:, None, :].repeat(n_t, 1), Covs), axis=2)
	alpha_inv_term = np.einsum(
		'stS->sS',
		dt[None, :, None]*np.concatenate(
			(
				eye_Covs_aug,
				np.concatenate((Covs.transpose(2,1,0), np.einsum('atc,atC->ctC', Covs, Covs)), axis=2)
			),
			axis=0
		)
	)
	alpha_solve_term = np.einsum(
		'ats,at->s',
		eye_Covs_aug,
		dS
	)
	try:
		alpha_hat = np.linalg.solve(alpha_inv_term, alpha_solve_term)
	except np.linalg.linalg.LinAlgError:
		alpha_hat = np.linalg.solve(add_random_noise_to_Cov_mat(alpha_inv_term), alpha_solve_term)

	scaled_alpha = np.einsum('s,ats,t->at', alpha_hat, eye_Covs_aug, dt)
	err = dS - scaled_alpha
	sigma_tilde = np.einsum('at,At->aA', err/ dt[None, :], err)/dt.shape[0]
	return alpha_hat[:n_a], sigma_tilde, alpha_hat[n_a:]