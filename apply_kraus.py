import numpy as np


def apply_kraus(kraus_ops, rho):
    if not kraus_ops:
        raise ValueError("At least one Kraus operator is required")

    dim = kraus_ops[0].shape[0]
    result = np.zeros((dim, dim), dtype=np.complex128)
    for op in kraus_ops:
        result += op @ rho @ op.conj().T
    return result
