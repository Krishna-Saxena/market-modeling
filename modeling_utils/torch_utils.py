import numpy as np
import torch

FLOAT_TYPE = torch.float32
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_zero_grad_hook(mask):
	def hook(grad):
		return grad * mask

	return hook

def diag_mask_gen_fn(A):
	return torch.eye(A.shape[0])

def triu_mask_gen_fn(A):
	return torch.triu(torch.ones_like(A))

def detach(A):
	return A.detach().cpu().numpy().astype(np.float32)

def make_Tensor(A):
	if type(A) != torch.Tensor:
		A = torch.tensor(A, dtype=FLOAT_TYPE, device=DEVICE)
	return A

def make_param_Tensor(A):
	if type(A) != torch.Tensor:
		A = torch.tensor(A, dtype=FLOAT_TYPE, device=DEVICE, requires_grad=True)
	A.requires_grad_(True)
	return A

def get_dep_SRW_nll_loss(dS, dt, drift, Sigma):
	scaled_drift = torch.einsum('a,t->at', drift, dt)
	err = dS - scaled_drift

	quad_term = torch.einsum('at,at->t', err, torch.linalg.solve_ex(Sigma, err).result)
	scaled_quad_term = quad_term/dt

	return torch.sum(scaled_quad_term) / 2*dt.shape[0] + torch.logdet(Sigma)/2

def get_dep_Cov_SRW_nll_loss(dS, dt, Covs, drift, Sigma, theta):
	scaled_drift = torch.einsum('a,t->at', drift, dt) + torch.einsum('C,atC,t->at', theta, Covs, dt)
	err = dS - scaled_drift

	quad_term = torch.einsum('at,at->t', err, torch.linalg.solve_ex(Sigma, err).result)
	scaled_quad_term = torch.einsum('a,t->a', quad_term, 1./dt)

	return torch.sum(scaled_quad_term) / 2*dt.shape[0] + torch.logdet(Sigma)/2