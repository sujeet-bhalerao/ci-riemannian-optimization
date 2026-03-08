

import numpy as np

from pymanopt import Problem, function
from pymanopt.manifolds.euclidean import Euclidean
from pymanopt.manifolds.product import Product
from pymanopt.optimizers import ConjugateGradient

from coherent_cost_full import coherent_cost_full
from gradient_full import gradient_full
from manifolds import ComplexStiefel
from tensor_kraus import tensor_kraus
from utils import normalize_density, softmax


def _prepare_operators(ops):
    prepared = []
    for op in ops:
        prepared.append(np.asarray(op, dtype=np.complex128))
    return tuple(prepared)


def run_opt_full(
    K,
    Kc,
    n,
    k,
    num_starts=30,
    eps_reg=1e-9,
    tolgradnorm=1e-8,
    maxiter=10_000,
    seed=None,
    verbose=False,
):
    K = _prepare_operators(K)
    Kc = _prepare_operators(Kc)

    if n > 1:
        K, Kc = tensor_kraus(K, Kc, n)

    dim = K[0].shape[1]
    if k > dim:
        raise ValueError("Rank k cannot be larger than dimension d")

    stiefel = ComplexStiefel(dim, k)
    euclidean = Euclidean(k)
    manifold = Product((stiefel, euclidean))

    @function.numpy(manifold)
    def cost(U, a):
        return coherent_cost_full(U, a, K, Kc, eps_reg)

    @function.numpy(manifold)
    def egrad(U, a):
        return gradient_full(U, a, K, Kc, eps_reg)

    problem = Problem(manifold=manifold, cost=cost, euclidean_gradient=egrad)
    solver = ConjugateGradient(
        max_iterations=maxiter,
        min_gradient_norm=tolgradnorm,
        verbosity=0,
    )

    rng = np.random.default_rng(seed)
    best_h = -np.inf
    best_rho = None
    best_lambda = None

    for _ in range(num_starts):
        random_matrix = rng.normal(size=(dim, k)) + 1j * rng.normal(size=(dim, k))
        U0, _ = np.linalg.qr(random_matrix, mode="reduced")
        a0 = rng.normal(size=k)

        result = solver.run(problem, initial_point=(U0, a0))
        candidate_U, candidate_a = result.point
        lambdas = softmax(candidate_a)
        rho = (
            (candidate_U * lambdas) @ candidate_U.conj().T
            + eps_reg * np.eye(dim, dtype=np.complex128)
        ) / (1.0 + eps_reg * dim)
        rho = normalize_density(rho)
        if verbose:
            print("  Current best h =", best_h / n)
            print("  Current h =", -result.cost)
        h_value = -result.cost
        if h_value > best_h:
            best_h = h_value
            best_rho = rho
            best_lambda = lambdas.astype(np.float64)

    if best_rho is None or best_lambda is None:
        raise RuntimeError("Optimization failed to find a solution")

    return best_h, best_rho, best_lambda
