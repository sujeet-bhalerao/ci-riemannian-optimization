# Riemannian Optimization for Coherent Information

This repository implements an alternate Riemannian optimization approach to that in [2] for numerical evaluation of coherent information of multiple copies of quantum channels.

The code works with channels in Kraus form, builds tensor powers of the channel and its complementary channel, and optimizes a low rank density matrix ansatz with `pymanopt`.

The search state is parameterized as `rho = U diag(lambda) U*`, where `U` is on a complex Stiefel manifold and `lambda` is a probability vector. In the implementation, the optimizer works with the pair `(U, a)`, where `a` is an unconstrained real vector and `lambda = softmax(a)`. The gradient uses the same adjoint-channel and matrix-log terms that appear inside the Tehrani and Pereira [1] gradient formula, but this is not a direct implementation of their positive definite density manifold formulation.

Instead of optimizing directly over the manifold of positive definite matrices `D++`, this code optimizes a low-rank parameterization, with `k` acting as a rank parameter for the ansatz. A small `eps I` regularization is added before evaluating entropies and matrix logarithms, so the objective is computed on a nearby full-rank state for numerical stability and so that the log-based gradient is well behaved. It is also different from the Zhu, Mao, Fang, and Wang lower bound approach [2], which uses an interleaved local unitary code state ansatz on a product of unitary manifolds.

## Install

```bash
git clone https://github.com/sujeet-bhalerao/ci-riemannian-optimization.git
cd ci-riemannian-optimization
uv sync
```

## Run an example

```bash
uv run example_full.py
```

This example script prints the best coherent information value it found for the damping-dephasing channel, the optimized `lambda` vector, and the resulting density matrix. Here `lambda` is the vector of mixture weights in `rho = U diag(lambda) U*`, so its entries are nonnegative and add up to 1. This reproduces the result from Table 10 in [2] for $n = 3$ copies of the damping-dephasing channel, and $|R| = 2$. Changing $n$ to $4$ or $5$ and |R| to $3$ reproduces other entries in Table 10.

## Usage

Inputs

- `K` is a list of Kraus operators for the channel
- `Kc` is a list of Kraus operators for the complementary channel
- `n` is the number of channel uses
- `k` is the rank parameter for the search state

```python
from comp_kraus import comp_kraus
from run_opt_full import run_opt_full

K = [...]  # Kraus operators
Kc = comp_kraus(K, dims=(d_in, d_out))

value, rho, lambdas = run_opt_full(
    K,
    Kc,
    n=3,
    k=2,
    num_starts=30,
    maxiter=10_000,
    seed=0,
)

print("n copy coherent information =", value)
print("per use =", value / 3)
print("lambda =", lambdas)
```

## Files

- `example_full.py` runs a complete example on a small qubit channel
- `run_opt_full.py` runs the multi start optimization on the Stiefel plus simplex parameterization
- `coherent_cost_full.py` evaluates the coherent information objective as a minimization cost
- `gradient_full.py` computes gradients for `U` and the `lambda` parameterization
- `manifolds.py` defines the custom complex Stiefel manifold used by `pymanopt`
- `tensor_kraus.py`, `apply_kraus.py`, `adjoint_kraus.py`, and `comp_kraus.py` implement Kraus channel operations
- `von_neumann.py` and `utils.py` provide entropy and matrix helper functions

## Notes

This code explicitly builds tensor product Kraus operators for `n` copies, so memory use and runtime grow quickly as `n` increases. Up to $n = 5$ copies can be run on a laptop for qubit channels.

The optimizer uses a small `eps` regularization before matrix logarithms are evaluated. This improves numerical stability near the boundary of the state space and slightly perturbs the objective.

## References

[1] A. Tehrani and R. Pereira, "The coherent information on the manifold of positive definite density matrices," Journal of Mathematical Physics 62, 042201 (2021). https://doi.org/10.1063/5.0020254

[2] C. Zhu, H. Mao, K. Fang, and X. Wang, "Geometric optimization for quantum communication," arXiv:2509.15106 (2025). https://arxiv.org/abs/2509.15106
