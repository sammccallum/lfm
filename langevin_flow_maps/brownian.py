from fractions import Fraction
from math import factorial

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr


def _chen_coefficients(degree):
    # c[n, m] = (-1)^n (2m+1) sum_{k=m}^{n} (-1/2)^k (n+k)! / [(n-k)!(k-m)!(k+m+1)!]
    # for 0 <= m <= n, else 0. Summed in exact rationals: the alternating terms
    # otherwise cancel catastrophically in floating point.
    c = [[0.0] * degree for _ in range(degree)]
    for n in range(degree):
        for m in range(n + 1):
            total = sum(
                Fraction(-1, 2) ** k
                * Fraction(
                    factorial(n + k),
                    factorial(n - k) * factorial(k - m) * factorial(k + m + 1),
                )
                for k in range(m, n + 1)
            )
            c[n][m] = float((-1) ** n * (2 * m + 1) * total)
    return jnp.array(c)


class BrownianPolynomial(eqx.Module):
    degree: int
    n_array: jax.Array
    chen_coeff: jax.Array
    shape: tuple

    def __init__(self, degree, shape):
        self.degree = degree
        self.n_array = jnp.arange(self.degree)
        self.chen_coeff = _chen_coefficients(degree)
        self.shape = shape

    def sample(self, s, t, key):
        eps = jr.normal(key, shape=(self.degree, *self.shape))
        std = jnp.sqrt((t - s) / (2 * self.n_array + 1))
        return jnp.einsum("i,i...->i...", std, eps)

    def sample_path(self, s, t, n, key):
        assert n > 0 and (n & (n - 1)) == 0, "n must be a power of two"
        h = (t - s) / n
        eps = jr.normal(key, shape=(n, self.degree, *self.shape))
        std = jnp.sqrt(h / (2 * self.n_array + 1))
        return jnp.einsum("d,nd...->nd...", std, eps)

    def combine(self, I_su, I_ut):
        c = self.chen_coeff
        signs = (-1.0) ** (self.n_array[:, None] + self.n_array[None, :])
        left = jnp.einsum("nm,m...->n...", c, I_su, precision="highest")
        right = jnp.einsum("nm,m...->n...", c * signs, I_ut, precision="highest")
        return left + right

    def _combine_pairwise(self, coeffs):
        return jax.vmap(self.combine)(coeffs[0::2], coeffs[1::2])

    def combine_to_level(self, coeffs, level=0):
        n = coeffs.shape[0]
        assert n > 0 and (n & (n - 1)) == 0, "sequence length must be a power of two"
        max_level = n.bit_length() - 1  # n == 2 ** max_level
        assert 0 <= level <= max_level, f"level must be in [0, {max_level}]"
        for _ in range(max_level - level):
            coeffs = self._combine_pairwise(coeffs)
        return coeffs
