from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from langevin_flow_maps.transformer import LFMTransformer


class EulerMaruyamaFlowMap(eqx.Module):
    network: LFMTransformer
    atomic_numbers: jax.Array | None
    masses: jax.Array | None
    gamma: float = 1.0
    kT: float = 1.0

    @property
    def sigma(self):
        return jnp.sqrt(2.0 * self.gamma * self.kT / self.masses[:, None])

    def __call__(
        self,
        x0,
        v0,
        h,
        I_st,
        atomic_numbers=None,
        masses=None,
        atom_mask=None,
    ):
        # x0, v0: (N, 3), h: scalar, I_st: (degree, N, 3) -> (x1, v1).
        atomic_numbers = (
            self.atomic_numbers if atomic_numbers is None else atomic_numbers
        )
        masses = self.masses if masses is None else masses

        use_mask = atom_mask is not None
        if not use_mask:
            atom_mask = jnp.ones((x0.shape[0],), dtype=bool)
        else:
            atom_mask = jnp.asarray(atom_mask, dtype=bool)
        safe_masses = jnp.where(atom_mask, masses, 1.0)
        if use_mask:
            mean_v, mean_f, _ = self.network(
                x0, atomic_numbers, v0, safe_masses, h, I_st, atom_mask
            )
        else:
            mean_v, mean_f, _ = self.network(
                x0, atomic_numbers, v0, safe_masses, h, I_st
            )
        sigma = (
            jnp.sqrt(2.0 * self.gamma * self.kT / safe_masses[:, None])
            * atom_mask[:, None]
        )
        x1 = x0 + h * mean_v
        v1 = (
            v0
            + h * (-self.gamma * mean_v + mean_f / safe_masses[:, None])
            + sigma * I_st[0]
        )
        return (
            jnp.where(atom_mask[:, None], x1, 0.0),
            jnp.where(atom_mask[:, None], v1, 0.0),
        )

    def step(
        self,
        x,
        v,
        h,
        I_st,
        atomic_numbers=None,
        masses=None,
        atom_mask=None,
    ):
        return self(x, v, h, I_st, atomic_numbers, masses, atom_mask)

    def batch_step(
        self,
        x,
        v,
        h,
        I_st,
        atomic_numbers=None,
        masses=None,
        atom_mask=None,
    ):
        def batch_axis(value):
            return 0 if value is not None and value.ndim == 2 else None

        return eqx.filter_vmap(
            self,
            in_axes=(
                0,
                0,
                None,
                0,
                batch_axis(atomic_numbers),
                batch_axis(masses),
                batch_axis(atom_mask),
            ),
        )(x, v, h, I_st, atomic_numbers, masses, atom_mask)


class TemperatureCorrectedFlowMap(eqx.Module):
    base: EulerMaruyamaFlowMap
    group_index: jax.Array
    group_labels: jax.Array

    def __init__(self, base: EulerMaruyamaFlowMap, group_index, group_labels):
        group_index = np.asarray(group_index, dtype=np.int32)
        group_labels = np.asarray(group_labels, dtype=np.int32)
        if group_index.shape != np.asarray(base.masses).shape:
            raise ValueError("group_index must contain one entry per atom")
        if group_labels.ndim != 1 or group_labels.size == 0:
            raise ValueError("group_labels must be a non-empty one-dimensional array")
        if np.any(group_index < 0) or np.any(group_index >= group_labels.size):
            raise ValueError("group_index contains an invalid group")

        self.base = base
        self.group_index = jnp.asarray(group_index)
        self.group_labels = jnp.asarray(group_labels)

    @classmethod
    def from_model(
        cls,
        base: EulerMaruyamaFlowMap,
        group_by: Literal["element", "atom"] = "element",
    ) -> "TemperatureCorrectedFlowMap":
        atomic_numbers = np.asarray(base.atomic_numbers, dtype=np.int32)
        if group_by == "element":
            group_labels, group_index = np.unique(atomic_numbers, return_inverse=True)
        elif group_by == "atom":
            group_labels = np.arange(atomic_numbers.size, dtype=np.int32)
            group_index = group_labels
        else:
            raise ValueError("group_by must be 'element' or 'atom'")
        return cls(base, group_index, group_labels)

    def batch_step(self, x, v, h, I_st):
        x_hat, v_hat = self.base.batch_step(x, v, h, I_st)
        noise = self.base.sigma[None, :, :] * I_st[:, 0]
        mass = self.base.masses[None, :, None]

        atom_A = jnp.mean(jnp.sum(mass * v_hat**2, axis=-1), axis=0)
        atom_B = jnp.mean(jnp.sum(mass * v_hat * noise, axis=-1), axis=0)
        atom_C = jnp.mean(jnp.sum(mass * noise**2, axis=-1), axis=0)
        n_groups = self.group_labels.shape[0]
        A, B, C = (
            jax.ops.segment_sum(value, self.group_index, num_segments=n_groups)
            for value in (atom_A, atom_B, atom_C)
        )

        group_sizes = jax.ops.segment_sum(
            jnp.ones_like(self.base.masses), self.group_index, num_segments=n_groups
        )
        R = 3.0 * group_sizes * self.base.kT
        D = R - A
        discriminant = B**2 + C * D
        safe_C = jnp.where(C > 0.0, C, 1.0)
        root = jnp.sqrt(jnp.maximum(discriminant, 0.0))
        amplitudes = jnp.stack(((-B + root) / safe_C, (-B - root) / safe_C))
        nonnegative_noise_scale = amplitudes >= -1.0
        scores = jnp.where(nonnegative_noise_scale, jnp.abs(amplitudes), jnp.inf)
        amplitude = jnp.take_along_axis(
            amplitudes,
            jnp.argmin(scores, axis=0)[None, :],
            axis=0,
        )[0]
        feasible = (
            (C > 0.0) & (discriminant >= 0.0) & jnp.isfinite(jnp.min(scores, axis=0))
        )

        safe_A = jnp.where(A > 0.0, A, R)
        scale = jnp.sqrt(R / safe_A)
        atom_amplitude = jnp.where(feasible, amplitude, 0.0)[self.group_index]
        atom_scale = jnp.where(feasible, 1.0, scale)[self.group_index]
        v_corrected = (
            atom_scale[None, :, None] * v_hat + atom_amplitude[None, :, None] * noise
        )
        return x_hat, jnp.where(h > 0.0, v_corrected, v_hat)
