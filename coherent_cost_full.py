import numpy as np

from apply_kraus import apply_kraus
from utils import softmax
from von_neumann import von_neumann


def coherent_cost_full(U, a, K, Kc, eps):
    lambdas = softmax(a)
    dim = U.shape[0]
    rho = (U * lambdas) @ U.conj().T
    rho = (rho + eps * np.eye(dim, dtype=np.complex128)) / (1.0 + eps * dim)
    Nrho = apply_kraus(K, rho)
    Ncrho = apply_kraus(Kc, rho)

    return -(von_neumann(Nrho) - von_neumann(Ncrho)) / np.log(2.0)
