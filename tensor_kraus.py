import numpy as np


def _kron_expand(left, right):
    expanded = []
    for a in left:
        for b in right:
            expanded.append(np.kron(a, b))
    return expanded


def tensor_kraus(K, Kc, n):
    if n < 1:
        raise ValueError("n must be at least 1")

    Kn = [np.array([[1.0]], dtype=np.complex128)]
    Kcn = [np.array([[1.0]], dtype=np.complex128)]

    for _ in range(n):
        Kn = _kron_expand(Kn, K)
        Kcn = _kron_expand(Kcn, Kc)

    return Kn, Kcn
