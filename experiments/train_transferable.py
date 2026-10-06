import json
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from ase import units

import wandb
from langevin_flow_maps.brownian import BrownianPolynomial
from langevin_flow_maps.datasets.many_peptides import (
    ManyPeptidesDataset,
    MolecularBatch,
    load_many_peptides,
)
from langevin_flow_maps.flow_map import EulerMaruyamaFlowMap
from langevin_flow_maps.optim import kimi_muon, wsd_schedule
from langevin_flow_maps.transformer import LFMTransformer


def ema_update(model, ema_model, decay):
    params, static = eqx.partition(model, eqx.is_inexact_array)
    ema_params, _ = eqx.partition(ema_model, eqx.is_inexact_array)
    ema_params = jax.tree.map(
        lambda e, p: decay * e + (1.0 - decay) * p, ema_params, params
    )
    return eqx.combine(ema_params, static)


def augment(key, *arrays):
    q = jr.normal(key, (arrays[0].shape[0], 4))
    q = q / jnp.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    rotations = jnp.stack(
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
    ).reshape(-1, 3, 3)
    return tuple(jnp.einsum("bij,baj->bai", rotations, array) for array in arrays)


def masked_atom_mean(values, atom_mask):
    per_sample = jnp.sum(values * atom_mask, axis=1) / jnp.maximum(
        jnp.sum(atom_mask, axis=1), 1
    )
    return jnp.sum(per_sample) / jnp.maximum(jnp.sum(jnp.any(atom_mask, axis=1)), 1)


def matching_loss_components(
    model, xs, vs, forces, atomic_numbers, masses, atom_mask, degree
):
    zero_I = jnp.zeros((degree, xs.shape[1], 3))

    def heads(x, v, z, mass, mask):
        mean_v, mean_f, _ = model.network(x, z, v, mass, 0.0, zero_I, mask)
        return mean_v, mean_f

    mean_v, mean_f = eqx.filter_vmap(heads)(xs, vs, atomic_numbers, masses, atom_mask)
    return (
        masked_atom_mean(jnp.sum((mean_f - forces) ** 2, axis=-1), atom_mask),
        masked_atom_mean(jnp.sum((mean_v - vs) ** 2, axis=-1), atom_mask),
    )


def _rollout(model, x0, v0, h, I_seq, atomic_numbers, masses, atom_mask):
    def body(state, noise):
        return model(*state, h, noise, atomic_numbers, masses, atom_mask), None

    return jax.lax.scan(body, (x0, v0), I_seq)[0]


def distillation_loss(
    model,
    ema_model,
    poly,
    xs,
    vs,
    atomic_numbers,
    masses,
    atom_mask,
    h_min,
    h_max,
    n_teacher,
    key,
):
    batch_size = xs.shape[0]
    k_h, k_path = jr.split(key, 2)
    hs = jnp.exp(
        jr.uniform(k_h, (batch_size,), minval=jnp.log(h_min), maxval=jnp.log(h_max))
    )
    keys = jr.split(k_path, batch_size)
    I_seq = eqx.filter_vmap(poly.sample_path, in_axes=(0, 0, None, 0))(
        jnp.zeros(batch_size), hs, n_teacher, keys
    )
    I_st = eqx.filter_vmap(poly.combine_to_level)(I_seq)[:, 0]
    x_t, v_t = eqx.filter_vmap(_rollout, in_axes=(None, 0, 0, 0, 0, 0, 0, 0))(
        ema_model, xs, vs, hs / n_teacher, I_seq, atomic_numbers, masses, atom_mask
    )
    x_t, v_t = jax.lax.stop_gradient(x_t), jax.lax.stop_gradient(v_t)

    def student(x, v, h, noise, z, mass, mask):
        return model(x, v, h, noise, z, mass, mask)

    x_s, v_s = eqx.filter_vmap(student)(
        xs, vs, hs, I_st, atomic_numbers, masses, atom_mask
    )
    state_error = jnp.sum((x_s - x_t) ** 2 + (v_s - v_t) ** 2, axis=-1)
    per_sample = jnp.sum(state_error * atom_mask, axis=1) / jnp.maximum(
        jnp.sum(atom_mask, axis=1), 1
    )
    return jnp.sum(per_sample / hs) / jnp.maximum(
        jnp.sum(jnp.any(atom_mask, axis=1)), 1
    )


def save_model(model, meta, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    model_path, meta_path = directory / "model.eqx", directory / "meta.json"
    eqx.tree_serialise_leaves(model_path, model)
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    return str(model_path), str(meta_path)


def load_model_dir(directory):
    directory = Path(directory)
    meta = json.loads((directory / "meta.json").read_text())
    model = EulerMaruyamaFlowMap(
        LFMTransformer(key=jr.key(meta["seed"]), **meta["hparams"]),
        None,
        None,
        gamma=meta["gamma"],
        kT=meta["kT"],
    )
    return eqx.tree_deserialise_leaves(directory / "model.eqx", model), meta


def make_training_step(
    optim,
    poly,
    *,
    kT,
    degree,
    h_min,
    h_max,
    n_teacher,
    eta,
    ema_beta,
):
    @eqx.filter_value_and_grad(has_aux=True)
    def loss_fn(model, ema_model, match, distill, key):
        k_match, k_distill, k_loss = jr.split(key, 3)
        prepared = []
        for batch, batch_key in ((match, k_match), (distill, k_distill)):
            k_rot, k_vel = jr.split(batch_key)
            positions, forces = augment(k_rot, batch.positions, batch.forces)
            std = jnp.sqrt(kT / batch.masses) * batch.atom_mask
            velocities = std[:, :, None] * jr.normal(k_vel, positions.shape)
            prepared.append((positions, velocities, forces))
        xs, vs, forces = prepared[0]
        force_loss, velocity_loss = matching_loss_components(
            model,
            xs,
            vs,
            forces,
            match.atomic_numbers,
            match.masses,
            match.atom_mask,
            degree,
        )
        matching = force_loss + velocity_loss
        xs, vs, _ = prepared[1]
        distillation = distillation_loss(
            model,
            ema_model,
            poly,
            xs,
            vs,
            distill.atomic_numbers,
            distill.masses,
            distill.atom_mask,
            h_min,
            h_max,
            n_teacher,
            k_loss,
        )
        loss = eta * matching + (1.0 - eta) * distillation
        return loss, {
            "match": matching,
            "match_force": force_loss,
            "match_velocity": velocity_loss,
            "distill": distillation,
        }

    @eqx.filter_jit
    def make_step(model, ema_model, opt_state, match, distill, key):
        key, step_key = jr.split(key)
        (loss, aux), grads = loss_fn(model, ema_model, match, distill, step_key)
        updates, opt_state = optim.update(
            grads,
            opt_state,
            eqx.filter(model, eqx.is_inexact_array),
        )
        model = eqx.apply_updates(model, updates)
        ema_model = ema_update(model, ema_model, ema_beta)
        return model, ema_model, opt_state, key, loss, aux

    return make_step


def main(
    shards=None,
    pdb_tar=None,
    cache_dir="data/many_peptides",
    revision="aa566c89f1d5c24aeb61378fb279e5f986c9b64f",
    topology_revision="1af9336878122eb1d62894fe2fb3ff4b801a3216",
    forcefield_files=("amber14-all.xml", "implicit/obc1.xml"),
    sequences=None,
    output_dir="artifacts/transferable",
    seed=1,
    batch_size=80,
    max_atoms=178,
    shuffle_buffer=1024,
    n_steps=100_000,
    peak_lr=1e-3,
    warmup_steps=2000,
    cooldown_steps=None,
    weight_decay=0.1,
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
    temperature=310.0,
    gamma_tau_fs=100.0,
    h_min_fs=0.01,
    h_max_fs=10.0,
    n_teacher=2,
    eta=0.25,
    ema_beta=0.99,
    max_grad_norm=10.0,
    wandb_project="Langevin Flow Maps",
    wandb_name=None,
    log_every=100,
):
    if cooldown_steps is None:
        cooldown_steps = n_steps // 5
    cfg = json.loads(json.dumps(dict(locals()), default=str))
    jax.config.update("jax_enable_x64", False)
    jax.config.update("jax_default_matmul_precision", "highest")
    if pdb_tar is None:
        dataset = load_many_peptides(
            cache_dir=cache_dir,
            shards=shards,
            sequences=sequences,
            revision=revision,
            topology_revision=topology_revision,
            forcefield_files=forcefield_files,
        )
    else:
        dataset = ManyPeptidesDataset(
            shards,
            pdb_tar,
            sequences=sequences,
            forcefield_files=forcefield_files,
        )
    loader = dataset.iter_batches(
        batch_size,
        max_atoms=max_atoms,
        shuffle_buffer=shuffle_buffer,
        seed=seed,
    )
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
    model_key, train_key = jr.split(jr.key(seed))
    model = EulerMaruyamaFlowMap(
        LFMTransformer(key=model_key, **hparams),
        None,
        None,
        gamma=gamma,
        kT=kT,
    )
    ema_model = model
    poly = BrownianPolynomial(degree=degree, shape=(max_atoms, 3))
    schedule = wsd_schedule(
        peak_lr=peak_lr,
        n_steps=n_steps,
        warmup_steps=warmup_steps,
        cooldown_steps=cooldown_steps,
    )
    optim = kimi_muon(
        model,
        schedule,
        peak_lr=peak_lr,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
    )
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
    make_step = make_training_step(
        optim,
        poly,
        kT=kT,
        degree=degree,
        h_min=h_min_fs * units.fs,
        h_max=h_max_fs * units.fs,
        n_teacher=n_teacher,
        eta=eta,
        ema_beta=ema_beta,
    )
    n_match = round(batch_size * eta)
    n_params = sum(
        leaf.size for leaf in jax.tree.leaves(eqx.filter(model, eqx.is_array))
    )
    meta = {
        "datasets": list(dataset.sequences),
        "hparams": hparams,
        "max_atoms": max_atoms,
        "gamma": gamma,
        "kT": float(kT),
        "seed": seed,
        "config": cfg,
        "data": dataset.metadata,
        "optimizer": "kimi_muonw_adamw",
    }
    run = None
    if wandb_project is not None:
        run = wandb.init(
            project=wandb_project,
            name=wandb_name,
            job_type="train-peptides",
            config={**cfg, "gamma": gamma, "kT": float(kT), "n_params": n_params},
        )
    print(
        f"model: {n_params:,} params | peptides: {len(dataset.sequences):,} | "
        f"batch: {batch_size} | steps: {n_steps:,}",
        flush=True,
    )
    start = time.time()
    atoms_seen = 0
    try:
        for step in range(n_steps):
            batch = next(loader)
            atoms_seen += int(batch.atom_mask.sum())
            match = MolecularBatch(*(jnp.asarray(a[:n_match]) for a in batch))
            distill = MolecularBatch(*(jnp.asarray(a[n_match:]) for a in batch))
            model, ema_model, opt_state, train_key, loss, aux = make_step(
                model,
                ema_model,
                opt_state,
                match,
                distill,
                train_key,
            )
            if (step + 1) % log_every == 0 or step + 1 == n_steps:
                metrics = {"loss": float(loss), **{k: float(v) for k, v in aux.items()}}
                metrics["lr"] = float(schedule(step))
                metrics["atoms_seen"] = atoms_seen
                print(
                    f"step {step + 1:7d} | loss {metrics['loss']:.6f} | "
                    f"match {metrics['match']:.6f} | distill {metrics['distill']:.6f} | "
                    f"{(time.time() - start) / 60:.1f} min",
                    flush=True,
                )
                if run is not None:
                    run.log(metrics, step=step + 1)
        meta.update(step=n_steps, frames_seen=n_steps * batch_size, **metrics)
        model_path, meta_path = save_model(ema_model, meta, output_dir)
        if run is not None:
            run.summary.update(metrics)
            artifact = wandb.Artifact(
                run.name, type="model", metadata={"step": n_steps}
            )
            artifact.add_file(model_path)
            artifact.add_file(meta_path)
            run.log_artifact(artifact).wait()
        print(f"Saved final model to {output_dir}", flush=True)
    finally:
        loader.close()
        if run is not None:
            run.finish()
    return ema_model, meta


if __name__ == "__main__":
    main()
