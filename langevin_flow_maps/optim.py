import math

import equinox as eqx
import jax
import jax.numpy as jnp
import optax


def wsd_schedule(*, peak_lr, n_steps, warmup_steps=2000, cooldown_steps=None):
    if cooldown_steps is None:
        cooldown_steps = n_steps // 5
    schedules = [optax.constant_schedule(peak_lr)]
    boundaries = []
    if warmup_steps:
        schedules.insert(0, optax.linear_schedule(peak_lr / 500, peak_lr, warmup_steps))
        boundaries.append(warmup_steps)
    if cooldown_steps:

        def cooldown(step):
            progress = (
                jnp.clip(step / (cooldown_steps - 1), 0.0, 1.0)
                if cooldown_steps > 1
                else 1.0
            )
            return peak_lr * (1.0 - jnp.sqrt(progress))

        schedules.append(cooldown)
        boundaries.append(n_steps - cooldown_steps)
    return optax.join_schedules(schedules, boundaries)


def muon_labels(model):
    def label(leaf):
        if isinstance(leaf, eqx.nn.Linear):
            labels = jax.tree.map(lambda _: "adam", leaf)
            return eqx.tree_at(lambda linear: linear.weight, labels, "muon")
        return "adam"

    labels = jax.tree.map(
        label,
        eqx.filter(model, eqx.is_inexact_array),
        is_leaf=lambda x: isinstance(x, eqx.nn.Linear),
    )
    if isinstance(model.network.readout, eqx.nn.Linear):
        return eqx.tree_at(lambda tree: tree.network.readout.weight, labels, "adam")
    for head in ("head_v", "head_f", "head_e"):
        labels = eqx.tree_at(
            lambda tree, head=head: (
                getattr(tree.network.readout, head).layers[-1].weight
            ),
            labels,
            "adam",
        )
    return labels


def kimi_muon(model, schedule, *, peak_lr, weight_decay=0.1, max_grad_norm=10.0):
    labels = muon_labels(model)

    def orthogonalize(matrix):
        x = matrix.astype(jnp.float32)
        x = x / (jnp.linalg.norm(x) + 1e-7)
        tall = x.shape[0] > x.shape[1]
        for _ in range(5):
            a = (
                jnp.matmul(x.T, x, precision="highest")
                if tall
                else jnp.matmul(x, x.T, precision="highest")
            )
            b = -4.7750 * a + 2.0315 * jnp.matmul(a, a, precision="highest")
            x = 3.4445 * x + (
                jnp.matmul(x, b, precision="highest")
                if tall
                else jnp.matmul(b, x, precision="highest")
            )
        return x.astype(matrix.dtype) * (0.2 * math.sqrt(max(matrix.shape)))

    # Batch equal-shaped matrices to avoid a separate series of kernels per leaf.
    def project(updates, params):
        del params
        leaves, treedef = jax.tree.flatten(updates)
        groups = {}
        for i, leaf in enumerate(leaves):
            groups.setdefault((leaf.shape, leaf.dtype), []).append(i)
        result = list(leaves)
        for indices in groups.values():
            values = jax.vmap(orthogonalize)(jnp.stack([leaves[i] for i in indices]))
            for j, i in enumerate(indices):
                result[i] = values[j]
        return jax.tree.unflatten(treedef, result)

    branches = optax.multi_transform(
        {
            "muon": optax.chain(
                optax.trace(decay=0.95, nesterov=True), optax.stateless(project)
            ),
            "adam": optax.scale_by_adam(b1=0.9, b2=0.95, eps=1e-8),
        },
        lambda _: labels,
    )

    def decay_init(params):
        return optax.ScaleByScheduleState(count=jnp.zeros([], jnp.int32))

    def decay_update(updates, state, params):
        rate = schedule(state.count)
        coefficient = weight_decay * rate / peak_lr
        updates = jax.tree.map(lambda u, p: u + coefficient * p, updates, params)
        return updates, optax.ScaleByScheduleState(optax.safe_increment(state.count))

    decay = optax.GradientTransformation(decay_init, decay_update)
    scale = optax.scale_by_learning_rate(schedule)
    transforms = (
        optax.clip_by_global_norm(max_grad_norm),
        branches,
        decay,
        scale,
    )
    return optax.chain(*transforms)
