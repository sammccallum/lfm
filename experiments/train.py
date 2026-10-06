import json
import os
import sys
import tempfile
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import wandb
from ase import units

from langevin_flow_maps import dataloader, load_aldp, load_md17
from langevin_flow_maps.brownian import BrownianPolynomial
from langevin_flow_maps.flow_map import (
    EulerMaruyamaFlowMap,
    TemperatureCorrectedFlowMap,
)
from langevin_flow_maps.transformer import LFMTransformer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def build_model(hparams, atomic_numbers, masses, gamma, kT, key):
    network = LFMTransformer(key=key, **hparams)
    return EulerMaruyamaFlowMap(
        network=network,
        atomic_numbers=atomic_numbers,
        masses=masses,
        gamma=gamma,
        kT=kT,
    )


def model_meta(dataset, hparams, atomic_numbers, masses, gamma, kT, seed):
    return {
        "dataset": dataset,
        "hparams": hparams,
        "atomic_numbers": atomic_numbers.tolist(),
        "masses": masses.tolist(),
        "gamma": gamma,
        "kT": float(kT),
        "seed": seed,
    }


def save_model(model, meta, directory):
    model_path = os.path.join(directory, "model.eqx")
    meta_path = os.path.join(directory, "meta.json")
    eqx.tree_serialise_leaves(model_path, model)
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)
    return model_path, meta_path


def load_model_dir(directory):
    with open(os.path.join(directory, "meta.json")) as fh:
        meta = json.load(fh)
    skeleton = build_model(
        meta["hparams"],
        jnp.asarray(meta["atomic_numbers"]),
        jnp.asarray(meta["masses"]),
        meta["gamma"],
        meta["kT"],
        jr.key(meta["seed"]),
    )
    model = eqx.tree_deserialise_leaves(os.path.join(directory, "model.eqx"), skeleton)
    return model, jnp.asarray(meta["atomic_numbers"]), jnp.asarray(meta["masses"]), meta


def ema_update(model, ema_model, decay):
    params, static = eqx.partition(model, eqx.is_inexact_array)
    ema_params, _ = eqx.partition(ema_model, eqx.is_inexact_array)
    ema_params = jax.tree.map(
        lambda e, p: decay * e + (1.0 - decay) * p, ema_params, params
    )
    return eqx.combine(ema_params, static)


def _random_rotations(key, n):
    q = jr.normal(key, (n, 4))
    q = q / jnp.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return jnp.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ],
        axis=-1,
    ).reshape(n, 3, 3)


def augment(key, *arrays):
    Q = _random_rotations(key, arrays[0].shape[0])
    return tuple(jnp.einsum("bij,baj->bai", Q, a) for a in arrays)


def sample_velocities(key, masses, batch_size, temperature, temperature_std):
    k_temp, k_vel = jr.split(key)
    temperatures = jnp.maximum(
        temperature + temperature_std * jr.normal(k_temp, (batch_size, 1, 1)),
        0.0,
    )
    std = jnp.sqrt(units.kB * temperatures / masses[None, :, None])
    return std * jr.normal(k_vel, (batch_size, masses.shape[0], 3))


def matching_loss(model, xs, vs, atomic_numbers, masses, degree, f_true):
    N = xs.shape[1]
    zero_I = jnp.zeros((degree, N, 3))

    def heads(x, v):
        mean_v, mean_f, _ = model.network(x, atomic_numbers, v, masses, 0.0, zero_I)
        return mean_v, mean_f

    mean_v, mean_f = eqx.filter_vmap(heads)(xs, vs)
    v_err = jnp.mean(jnp.sum((mean_v - vs) ** 2, axis=-1))
    f_err = jnp.mean(jnp.sum((mean_f - f_true) ** 2, axis=-1))
    return v_err + f_err


def distillation_loss(model, ema_model, poly, xs, vs, h_min, h_max, key):
    B = xs.shape[0]
    k_h, k_su, k_ut = jr.split(key, 3)

    hs = jnp.exp(jr.uniform(k_h, (B,), minval=jnp.log(h_min), maxval=jnp.log(h_max)))
    I_su = eqx.filter_vmap(poly.sample, in_axes=(None, 0, 0))(
        0.0, hs / 2, jr.split(k_su, B)
    )
    I_ut = eqx.filter_vmap(poly.sample, in_axes=(0, 0, 0))(
        hs / 2, hs, jr.split(k_ut, B)
    )
    I_st = eqx.filter_vmap(poly.combine)(I_su, I_ut)

    x_u, v_u = eqx.filter_vmap(ema_model)(xs, vs, hs / 2, I_su)
    x_t, v_t = eqx.filter_vmap(ema_model)(x_u, v_u, hs / 2, I_ut)
    x_t, v_t = jax.lax.stop_gradient(x_t), jax.lax.stop_gradient(v_t)

    x_s, v_s = eqx.filter_vmap(model, in_axes=(0, 0, 0, 0))(xs, vs, hs, I_st)

    per_sample = jnp.sum((x_s - x_t) ** 2, axis=(1, 2)) + jnp.sum(
        (v_s - v_t) ** 2, axis=(1, 2)
    )
    return jnp.mean(per_sample / hs)


def main(
    dataset="aspirin",
    seed=1,
    batch_size=512,
    n_steps=100_000,
    peak_lr=1e-3,
    warmup_steps=2_000,
    end_lr=1e-7,
    eval_every=2_000,
    val_subset=4_096,
    hidden_size=256,
    n_heads=8,
    n_blocks=6,
    num_basis=10,
    cutoff=7.5,
    num_vel_basis=8,
    vel_max=1.55,
    degree=4,
    num_noise_basis=8,
    noise_max=2.0,
    temperature=500.0,  # aldp: 300K, MD17: 500K
    velocity_temperature_std=0.0,
    gamma_tau_fs=100.0,
    h_min_fs=0.01,
    h_max_fs=10.0,
    eta=0.25,
    ema_beta=0.99,
    max_grad_norm=10.0,
    wandb_project="Langevin Flow Maps",
):
    cfg = dict(locals())
    hparams = dict(
        hidden_size=hidden_size,
        n_heads=n_heads,
        n_blocks=n_blocks,
        num_basis=num_basis,
        cutoff=cutoff,
        num_vel_basis=num_vel_basis,
        vel_max=vel_max,
        degree=degree,
        num_noise_basis=num_noise_basis,
        noise_max=noise_max,
    )
    kT = units.kB * temperature
    gamma = 1.0 / (gamma_tau_fs * units.fs)
    h_min, h_max = h_min_fs * units.fs, h_max_fs * units.fs

    if dataset == "aldp":
        train, val, test = load_aldp()
    else:
        train, val, test = load_md17(dataset)  # pyright: ignore
    atomic_numbers = jnp.asarray(train.atomic_numbers).reshape(-1)
    masses = jnp.asarray(train.masses).reshape(-1)
    n_atoms = atomic_numbers.shape[0]

    train_positions = jnp.asarray(train.positions)
    train_forces = jnp.asarray(train.forces)
    val_positions = jnp.asarray(val.positions)[:val_subset]
    val_forces = jnp.asarray(val.forces)[:val_subset]
    test_positions = jnp.asarray(test.positions)
    test_forces = jnp.asarray(test.forces)

    model_key, loader_key, train_key = jr.split(jr.key(seed), 3)
    model = build_model(hparams, atomic_numbers, masses, gamma, kT, model_key)
    ema_model = model
    poly = BrownianPolynomial(degree=degree, shape=(n_atoms, 3))
    n_params = sum(
        x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array))
    )
    print(
        f"model: {n_params:,} params  |  molecule: {atomic_numbers.tolist()}",
        flush=True,
    )

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=1e-6,
        peak_value=peak_lr,
        warmup_steps=warmup_steps,
        decay_steps=n_steps,
        end_value=end_lr,
    )
    optim = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adam(schedule, b1=0.9, b2=0.95),
    )
    filter_spec = jax.tree_util.tree_map(eqx.is_inexact_array, model)
    filter_spec = eqx.tree_at(lambda m: m.masses, filter_spec, replace=False)
    opt_state = optim.init(eqx.filter(model, filter_spec))

    n_match = int(round(batch_size * eta))

    @eqx.filter_value_and_grad(has_aux=True)
    def loss_fn(model, ema_model, xs, vs, f_true, key):
        matching = matching_loss(
            model,
            xs[:n_match],
            vs[:n_match],
            atomic_numbers,
            masses,
            degree,
            f_true[:n_match],
        )
        distillation = distillation_loss(
            model,
            ema_model,
            poly,
            xs[n_match:],
            vs[n_match:],
            h_min,
            h_max,
            key,
        )
        loss = eta * matching + (1.0 - eta) * distillation
        return loss, (matching, distillation)

    @eqx.filter_jit
    def make_step(model, ema_model, opt_state, positions, forces, key):
        k_rot, k_vel, k_loss = jr.split(key, 3)
        positions, forces = augment(k_rot, positions, forces)
        velocities = sample_velocities(
            k_vel, masses, batch_size, temperature, velocity_temperature_std
        )
        (loss, aux), grads = loss_fn(
            model, ema_model, positions, velocities, forces, k_loss
        )
        updates, opt_state = optim.update(
            eqx.filter(grads, filter_spec), opt_state, eqx.filter(model, filter_spec)
        )
        model = eqx.apply_updates(model, updates)
        ema_model = ema_update(model, ema_model, ema_beta)
        return model, ema_model, opt_state, loss, aux

    @eqx.filter_jit
    def force_mae(model, positions, forces):
        zero_v = jnp.zeros_like(positions)
        zero_I = jnp.zeros((degree, n_atoms, 3))

        def pred(x, v):
            return model.network(x, atomic_numbers, v, masses, 0.0, zero_I)[1]

        def body(carry, batch):
            p, v, f = batch
            pf = eqx.filter_vmap(pred)(p, v)
            return carry + jnp.sum(jnp.abs(pf - f)), None

        nb = positions.shape[0] // batch_size
        p = positions[: nb * batch_size].reshape(nb, batch_size, n_atoms, 3)
        v = zero_v[: nb * batch_size].reshape(nb, batch_size, n_atoms, 3)
        f = forces[: nb * batch_size].reshape(nb, batch_size, n_atoms, 3)
        ae, _ = jax.lax.scan(body, 0.0, (p, v, f))
        return ae / (nb * batch_size * n_atoms * 3)

    run = None
    if wandb_project is not None:
        run = wandb.init(
            project=wandb_project,
            job_type="train",
            config={**cfg, "gamma": gamma, "kT": float(kT), "n_params": n_params},
        )

    loader = dataloader(
        (train_positions, train_forces), batch_size=batch_size, key=loader_key
    )
    print(f"training for {n_steps:,} steps...", flush=True)
    start = time.time()
    for step, (positions, forces) in zip(range(1, n_steps + 1), loader):
        train_key, step_key = jr.split(train_key)
        model, ema_model, opt_state, loss, aux = make_step(
            model, ema_model, opt_state, positions, forces, step_key
        )

        if step % eval_every == 0:
            val_mae = force_mae(ema_model, val_positions, val_forces)
            elapsed = time.time() - start
            eta_min = elapsed / step * (n_steps - step) / 60
            if run is not None:
                run.log(
                    {
                        "loss": float(loss),
                        "match": float(aux[0]),
                        "distill": float(aux[1]),
                        "val_force_mae": float(val_mae),
                        "lr": float(jnp.asarray(schedule(step))),
                    },
                    step=step,
                )
            print(
                f"step {step:6d}/{n_steps}  loss {float(loss):8.4f}  "
                f"match {float(aux[0]):8.4f}  distill {float(aux[1]):8.4f}  "
                f"val force MAE {float(val_mae):7.4f} eV/Ang  "
                f"lr {float(jnp.asarray(schedule(step))):.2e}  [{elapsed / 60:5.1f}m, ETA {eta_min:4.1f}m]",
                flush=True,
            )

    test_mae = force_mae(ema_model, test_positions, test_forces)
    print(
        f"\ntest force (ema) MAE {float(test_mae):.4f} eV/Ang"
        f"\ntotal time {(time.time() - start) / 60:.1f} min",
        flush=True,
    )

    meta = model_meta(dataset, hparams, atomic_numbers, masses, gamma, kT, seed)
    if run is not None:
        run.summary["test_force_mae"] = float(test_mae)
        with tempfile.TemporaryDirectory() as d:
            model_path, meta_path = save_model(ema_model, meta, d)
            artifact = wandb.Artifact(
                run.name,  # pyright: ignore
                type="model",
                metadata={"dataset": dataset},
            )
            artifact.add_file(model_path)
            artifact.add_file(meta_path)
            run.log_artifact(artifact).wait()
        print(f"logged model artifact '{run.name}'", flush=True)

    from experiments.eval import evaluate

    evaluate(
        ema_model,
        atomic_numbers,
        masses,
        dataset,
        run,
        name="original",
    )
    corrected_model = TemperatureCorrectedFlowMap.from_model(ema_model)
    evaluate(
        corrected_model,
        atomic_numbers,
        masses,
        dataset,
        run,
        name="corrected",
    )
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
