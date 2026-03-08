import numpy as np


def softmax(values):
    values = np.asarray(values, dtype=np.float64)
    shifted = values - np.max(values)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values)


def hermitian_part(matrix):
    matrix = np.asarray(matrix, dtype=np.complex128)
    return 0.5 * (matrix + matrix.conj().T)


def normalize_density(matrix):
    hermitian = hermitian_part(matrix)
    trace = np.trace(hermitian)
    if not np.isclose(trace, 1.0):
        hermitian = hermitian / trace
    return hermitian
