import numpy as np
from scipy.linalg import logm

from adjoint_kraus import adjoint_kraus
from apply_kraus import apply_kraus
from utils import softmax


def gradient_full(U, a, K, Kc, eps):
    lambdas = softmax(a)
    dim = U.shape[0]

    rho = (U * lambdas) @ U.conj().T
    rho = (rho + eps * np.eye(dim, dtype=np.complex128)) / (1.0 + eps * dim)

    Nrho = apply_kraus(K, rho)
    Ncrho = apply_kraus(Kc, rho)

    L = adjoint_kraus(Kc, logm(Ncrho)) - adjoint_kraus(K, logm(Nrho))

    scale = np.log(2.0) * (1.0 + eps * dim)
    grad_U = -2.0 * (L @ U) * lambdas / scale

    diag_terms = -np.real(np.diag(U.conj().T @ L @ U)) / scale
    lambda_dot_grad = float(np.sum(lambdas * diag_terms))
    grad_a = lambdas * (diag_terms - lambda_dot_grad)

    return grad_U, grad_a.astype(np.float64)
