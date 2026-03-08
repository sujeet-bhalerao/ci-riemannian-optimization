import numpy as np

from pymanopt.manifolds.manifold import Manifold

from utils import hermitian_part


class ComplexStiefel(Manifold):
    def __init__(self, n, p):
        if n < p:
            raise ValueError("Require n >= p for ComplexStiefel")
        name = "Complex Stiefel manifold St(%d,%d)" % (n, p)
        dimension = int(2 * n * p - p * p)
        super().__init__(name=name, dimension=dimension, point_layout=1)
        self._n = n
        self._p = p

    @property
    def typical_dist(self):
        return float(np.sqrt(self._p))

    def inner_product(self, point, tangent_vector_a, tangent_vector_b):
        return float(np.real(np.vdot(tangent_vector_a, tangent_vector_b)))

    def projection(self, point, vector):
        correction = hermitian_part(point.conj().T @ vector)
        return vector - point @ correction

    to_tangent_space = projection

    def retraction(self, point, tangent_vector):
        y = point + tangent_vector
        q, r = np.linalg.qr(y, mode="reduced")
        diag = np.diag(r)
        phases = np.ones_like(diag, dtype=np.complex128)
        nonzero = np.abs(diag) > 0
        phases[nonzero] = diag[nonzero] / np.abs(diag[nonzero])
        return q * phases.conj()

    def random_point(self):
        real_part = np.random.normal(size=(self._n, self._p))
        imag_part = np.random.normal(size=(self._n, self._p))
        z = real_part + 1j * imag_part
        q, _ = np.linalg.qr(z, mode="reduced")
        return q

    def random_tangent_vector(self, point):
        real_part = np.random.normal(size=(self._n, self._p))
        imag_part = np.random.normal(size=(self._n, self._p))
        z = real_part + 1j * imag_part
        tangent = self.projection(point, z)
        norm = np.linalg.norm(tangent)
        if norm == 0:
            return self.zero_vector(point)
        return tangent / norm

    def transport(self, point_a, point_b, tangent_vector):
        return self.projection(point_b, tangent_vector)

    def norm(self, point, tangent_vector):
        return float(np.linalg.norm(tangent_vector))

    def zero_vector(self, point):
        return np.zeros_like(point, dtype=np.complex128)

    def euclidean_to_riemannian_gradient(self, point, euclidean_gradient):
        return self.projection(point, euclidean_gradient)

    def dist(self, point_a, point_b):
        return float(np.linalg.norm(point_a - point_b))
