import numpy as np

from utils import hermitian_part


def von_neumann(matrix):
    hermitian = hermitian_part(matrix)
    eigenvalues = np.linalg.eigvalsh(hermitian)
    positive = eigenvalues[eigenvalues > 0]
    if positive.size == 0:
        return 0.0
    entropy = -np.sum(positive * np.log(positive))
    return float(np.real(entropy))
