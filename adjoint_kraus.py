import numpy as np


def adjoint_kraus(kraus_ops, matrix):
    if not kraus_ops:
        raise ValueError("At least one Kraus operator is required")

    dim = kraus_ops[0].shape[1]
    result = np.zeros((dim, dim), dtype=np.complex128)
    for op in kraus_ops:
        result += op.conj().T @ matrix @ op
    return result
