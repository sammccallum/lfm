import os
import sys
import time

import ase
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
from ase import units
from ase.data import chemical_symbols
from ase.neighborlist import NeighborList, natural_cutoffs

import wandb
from langevin_flow_maps import load_aldp, load_md17
from langevin_flow_maps.brownian import BrownianPolynomial
from langevin_flow_maps.flow_map import (
    EulerMaruyamaFlowMap,
    TemperatureCorrectedFlowMap,
)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
from experiments.train import load_model_dir  # noqa: E402


def maxwell_boltzmann(key, masses, kT, shape):
    std = jnp.sqrt(kT / masses)[None, :, None]
    return std * jr.normal(key, shape)


def h0_force(model, x, v):
    zero_I = jnp.zeros((model.network.degree, model.masses.shape[0], 3))
    return model.network(x, model.atomic_numbers, v, model.masses, 0.0, zero_I)[1]


class BAOAB(eqx.Module):
    model: EulerMaruyamaFlowMap

    def __call__(self, x, v, h, I_st):
        # BAOAB computes the exact variance of the OU integral. However
        # given a realisation of the path W_t it does not provide a recipe
        # for approximating this integral **pathwise**. 
        # 
        # We therefore choose the first-order approximation using the
        # increment I_st[0] = W_t - W_s. This preserves the exact OU variance
        # and obtains first order stong convergence.
        m = self.model
        inv_m = (1.0 / m.masses)[:, None]
        c = jnp.exp(-m.gamma * h)
        d = jnp.sqrt(-jnp.expm1(-2.0 * m.gamma * h) / (2.0 * m.gamma * h))
        v = v + 0.5 * h * h0_force(m, x, v) * inv_m
        x = x + 0.5 * h * v
        v = c * v + m.sigma * d * I_st[0]
        x = x + 0.5 * h * v
        v = v + 0.5 * h * h0_force(m, x, v) * inv_m
        return x, v

    def batch_step(self, x, v, h, I_st):
        return eqx.filter_vmap(self, in_axes=(0, 0, None, 0))(x, v, h, I_st)


class EulerMaruyama(eqx.Module):
    model: EulerMaruyamaFlowMap

    def __call__(self, x, v, h, I_st):
        m = self.model
        force = h0_force(m, x, v)
        return (
            x + h * v,
            v + h * (-m.gamma * v + force / m.masses[:, None]) + m.sigma * I_st[0],
        )

    def batch_step(self, x, v, h, I_st):
        return eqx.filter_vmap(self, in_axes=(0, 0, None, 0))(x, v, h, I_st)


def schedule(dt_fs, total_ps, snap_fs):
    stride = max(1, round(snap_fs / dt_fs))
    n_snap = round(total_ps * 1000.0 / (stride * dt_fs))
    return n_snap, stride


@eqx.filter_jit
def roll(solver, poly, x0, v0, h, n_snap, stride, key):
    # Free roll of any (x, v, h, I_st) solver, snapshotting every `stride` steps.
    vsample = eqx.filter_vmap(poly.sample, in_axes=(None, None, 0))

    def inner(carry, key):
        x, v = carry
        coefficients = vsample(0.0, h, jr.split(key, x.shape[0]))
        return solver.batch_step(x, v, h, coefficients), None

    def outer(carry, key):
        carry, _ = jax.lax.scan(inner, carry, jr.split(key, stride))
        return carry, carry

    _, (xs, vs) = jax.lax.scan(outer, (x0, v0), jr.split(key, n_snap))
    return xs, vs


# --------------------------------------------------------------------------- #
# Strong and weak error vs reference BAOAB over shared Brownian paths
# --------------------------------------------------------------------------- #
def _is_power_of_two(n):
    return n > 0 and (n & (n - 1)) == 0


def sample_fine_paths(poly, n_traj, n_ref_steps, T, key):
    assert _is_power_of_two(n_ref_steps), "n_ref_steps must be a power of two"
    keys = jr.split(key, n_traj)
    return jax.vmap(lambda k: poly.sample_path(0.0, T, n_ref_steps, k))(keys)


def _solve(solver, x0, v0, h, I_seq):
    def body(carry, I_st):
        x, v = solver.batch_step(carry[0], carry[1], h, I_st)
        return (x, v), (x, v)

    time_first = jnp.swapaxes(I_seq, 0, 1)
    _, (xs, vs) = jax.lax.scan(body, (x0, v0), time_first)
    xs = jnp.concatenate([x0[None], xs], axis=0)
    vs = jnp.concatenate([v0[None], vs], axis=0)
    return jnp.swapaxes(xs, 0, 1), jnp.swapaxes(vs, 0, 1)


@eqx.filter_jit
def _reference_trajs(ref_solver, x0, v0, h_fine, I_fine):
    return _solve(ref_solver, x0, v0, h_fine, I_fine)


@eqx.filter_jit
def _candidate_stats(solver, poly, x0, v0, h, level, stride, I_fine, ref_xs, ref_vs):
    I_coarse = jax.vmap(lambda coeffs: poly.combine_to_level(coeffs, level))(I_fine)
    xs, vs = _solve(solver, x0, v0, h, I_coarse)
    rx, rv = ref_xs[:, ::stride], ref_vs[:, ::stride]
    mse = 0.5 * (
        jnp.mean((xs - rx) ** 2, axis=(1, 2, 3))
        + jnp.mean((vs - rv) ** 2, axis=(1, 2, 3))
    )
    return mse, xs[:, -1], vs[:, -1]


def weak_error_with_se(candidate, reference):
    delta = (candidate - reference).reshape(candidate.shape[0], -1)
    mean_delta = jnp.mean(delta, axis=0)
    error = jnp.mean(jnp.abs(mean_delta))
    if delta.shape[0] < 2:
        return float(error), float("nan")
    leave_one_out = (delta.shape[0] * mean_delta[None, :] - delta) / (
        delta.shape[0] - 1
    )
    estimates = jnp.mean(jnp.abs(leave_one_out), axis=1)
    se = jnp.sqrt(
        (delta.shape[0] - 1) * jnp.mean((estimates - jnp.mean(estimates)) ** 2)
    )
    return float(error), float(se)


def error_sweep(solver, poly, x0, v0, T, I_fine, ref_xs, ref_vs, n_steps_list):
    # Each candidate rolled over the shared fine path Chen-aggregated down to n_steps
    # coarse intervals. Strong error: per-path RMSE at the shared grid points. Weak
    # error: |E[.]_h - E[.]_ref| of the terminal state, position and velocity separately.
    # Sharing the Brownian path (and initial condition) cancels the common part in the
    # mean, leaving the O(h^p) bias with far less Monte-Carlo noise.
    n_ref = I_fine.shape[1]
    ref_xT, ref_vT = ref_xs[:, -1], ref_vs[:, -1]
    results = []
    for n_steps in n_steps_list:
        assert _is_power_of_two(n_steps), "each n_steps must be a power of two"
        assert n_steps <= n_ref, "n_steps must not exceed n_ref_steps"
        level = n_steps.bit_length() - 1
        stride = n_ref // n_steps
        mse, xT, vT = _candidate_stats(
            solver, poly, x0, v0, T / n_steps, level, stride, I_fine, ref_xs, ref_vs
        )
        rmse = float(jnp.sqrt(jnp.mean(mse)))
        std = float(jnp.std(jnp.sqrt(mse)))
        wx, wx_se = weak_error_with_se(xT, ref_xT)
        wv, wv_se = weak_error_with_se(vT, ref_vT)
        results.append((n_steps, T / n_steps, rmse, std, wx, wv, wx_se, wv_se))
    return results


def fit_order(hs, errs):
    hs, errs = np.asarray(hs, float), np.asarray(errs, float)
    m = errs > 0
    return float(np.polyfit(np.log(hs[m]), np.log(errs[m]), 1)[0])


def strong_order(results):
    return fit_order([r[1] for r in results], [r[2] for r in results])


def weak_orders(results):
    hs = [r[1] for r in results]
    return fit_order(hs, [r[4] for r in results]), fit_order(
        hs, [r[5] for r in results]
    )


def _draw_strong(ax, fm_results, em_results, baoab_results, fm_label="flow map"):
    for results, label, fmt in (
        (fm_results, fm_label, "o-"),
        (em_results, "Euler-Maruyama", "^-."),
        (baoab_results, "BAOAB", "s--"),
    ):
        hs = np.array([r[1] / units.fs for r in results])
        rmse = np.array([r[2] for r in results])
        std = np.array([r[3] for r in results])
        ax.errorbar(hs, rmse, yerr=std, fmt=fmt, capsize=3, label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("step size h (fs)")
    ax.set_ylabel("strong RMSE")
    ax.set_title("Strong convergence vs reference BAOAB")
    ax.legend()
    ax.grid(alpha=0.3, which="both")


def _draw_weak(ax, fm_results, baoab_results, fm_label="flow map"):
    for results, label, marker in (
        (fm_results, fm_label, "o"),
        (baoab_results, "BAOAB", "s"),
    ):
        hs = np.array([r[1] / units.fs for r in results])
        wx = np.array([r[4] for r in results])
        wv = np.array([r[5] for r in results])
        ox, ov = weak_orders(results)
        ax.plot(hs, wx, marker + "-", label=f"{label} pos ({ox:.2f})")
        ax.plot(hs, wv, marker + "--", alpha=0.7, label=f"{label} vel ({ov:.2f})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("step size h (fs)")
    ax.set_ylabel("weak error  |E[.]$_h$ - E[.]$_{ref}$|")
    ax.set_title("Weak convergence (terminal moment) vs reference BAOAB")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")


# --------------------------------------------------------------------------- #
# Distributional metrics: one rollout -> h(r) MAE, kinetic deviation
# --------------------------------------------------------------------------- #
def bond_indices(atomic_numbers, ref_frame):
    atoms = ase.Atoms(
        positions=np.asarray(ref_frame),
        numbers=np.asarray(atomic_numbers).reshape(-1),
    )
    nl = NeighborList(natural_cutoffs(atoms), self_interaction=False)
    nl.update(atoms)
    i, j = nl.get_connectivity_matrix().todense().nonzero()  # pyright: ignore
    return np.asarray(i), np.asarray(j)


def bond_lengths(traj, bonds):
    i, j = bonds
    d = traj[..., i, :] - traj[..., j, :]
    return jnp.sqrt(jnp.sum(d**2, axis=-1))


def collapse_points(xs, bonds, ref_mean, threshold):
    # Per trajectory, the first snapshot at which any bond leaves threshold of its
    # reference mean length; xs.shape[0] (never) if it holds for the whole roll.
    bl = bond_lengths(jnp.swapaxes(xs, 0, 1), bonds)  # (B, n_snap, n_bonds)
    dev = jnp.max(jnp.abs(bl - ref_mean), axis=-1)
    exceed = dev > threshold
    return jnp.where(exceed.any(axis=1), jnp.argmax(exceed, axis=1), xs.shape[0])


def pairwise_distances(frames):
    # Exact zeros (self-pairs) sit below any positive bins[0] and drop out.
    d = frames[:, :, None, :] - frames[:, None, :, :]
    return jnp.sqrt(jnp.sum(d**2, axis=-1)).reshape(-1)


def hist_density(items, feature, bins, chunk=4096):
    # Density of feature(items) over bins, accumulated in chunks to bound memory.
    # Values outside [bins[0], bins[-1]] drop out, as in np.histogram.
    counts = jnp.zeros(bins.shape[0] - 1)
    for i in range(0, items.shape[0], chunk):
        c, _ = jnp.histogram(feature(items[i : i + chunk]), bins=bins)
        counts = counts + c
    width = bins[1:] - bins[:-1]
    return counts / (counts.sum() * width)


def density_mae(sim, ref, scale):
    return float(jnp.abs(sim - ref).mean() * scale)


def kinetic_deviation(vel, masses, atomic_numbers, kT):
    # Per-atom kinetic temperature <m v^2 / kT> * T_target, averaged over frames and
    # the 3 components, grouped by element. Deviation from the target temperature.
    T_target = kT / units.kB
    per_atom = np.asarray(jnp.mean(masses[None, :, None] * vel**2 / kT, axis=(0, 2)))
    T_atom = per_atom * T_target
    Z = np.asarray(atomic_numbers)
    overall = float(T_atom.mean()) - T_target
    per_el = {int(z): float(T_atom[Z == z].mean()) - T_target for z in np.unique(Z)}
    return overall, per_el, T_target


def evaluate_rollout(
    model,
    poly,
    x0,
    v0,
    dt_fs,
    sim_ps,
    snap_fs,
    threshold,
    atomic_numbers,
    masses,
    bonds,
    ref_mean,
    bins,
    xlim,
    h_ref,
    kT,
    key,
):
    n_snap, stride = schedule(dt_fs, sim_ps, snap_fs)
    xs, vs = roll(model, poly, x0, v0, dt_fs * units.fs, n_snap, stride, key)
    jax.block_until_ready((xs, vs))
    collapse = collapse_points(xs, bonds, ref_mean, threshold)
    valid = jnp.arange(n_snap)[:, None] < collapse[None, :]
    sim_pos = xs[valid].reshape(-1, masses.shape[0], 3)
    sim_vel = vs[valid]

    collapse = np.asarray(collapse)
    survived = np.where(collapse >= n_snap, sim_ps, collapse * snap_fs / 1000.0)
    h_sim = hist_density(sim_pos, pairwise_distances, bins)
    dT, dT_el, T_target = kinetic_deviation(sim_vel, masses, atomic_numbers, kT)
    return dict(
        dt_fs=dt_fs,
        frames=sim_pos.shape[0],
        survival=float(survived.mean()),
        collapsed=int((collapse < n_snap).sum()),
        mae=density_mae(h_sim, h_ref, xlim),
        dT=dT,
        dT_el=dT_el,
        T_target=T_target,
        h_sim=h_sim,
    )


def plot_summary(
    bins, h_ref, rows, fm_results, em_results, baoab_results, fm_label="flow map"
):
    centers = np.asarray(0.5 * (bins[1:] + bins[:-1]))
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()

    _draw_strong(axes[0], fm_results, em_results, baoab_results, fm_label=fm_label)

    ax = axes[1]
    ax.plot(centers, np.asarray(h_ref), "k--", lw=2, label="reference (MD17)")
    for r in rows:
        ax.plot(
            centers,
            np.asarray(r["h_sim"]),
            lw=1.5,
            alpha=0.8,
            label=f"{r['dt_fs']:g} fs ({r['mae']:.4f})",
        )
    ax.set_xlim(0, 5)
    ax.set_xlabel("r (Ang)")
    ax.set_ylabel("h(r)")
    ax.set_title("Interatomic distance h(r) | MAE")
    ax.legend(fontsize=8)

    _draw_weak(axes[2], fm_results, baoab_results, fm_label=fm_label)

    ax = axes[3]
    hs = [r["dt_fs"] for r in rows]
    ax.axhline(0, color="0.7", lw=1)
    ax.plot(hs, [r["dT"] for r in rows], "ko-", lw=2, label="overall")
    for z in sorted(rows[0]["dT_el"]):
        ax.plot(
            hs,
            [r["dT_el"][z] for r in rows],
            "o-",
            alpha=0.8,
            label=chemical_symbols[z],
        )
    ax.set_xlabel("step size h (fs)")
    ax.set_ylabel("kinetic dT (K)")
    ax.set_title(f"Kinetic temperature deviation (target {rows[0]['T_target']:.0f} K)")
    ax.legend(fontsize=8)

    fig.tight_layout()
    return fig


def print_table(title, header, rows):
    print(f"\n{title}")
    print("  " + "  ".join(f"{h:>9}" for h in header))
    for row in rows:
        print("  " + "  ".join(f"{c:>9}" for c in row))


def evaluate(
    model: EulerMaruyamaFlowMap | TemperatureCorrectedFlowMap,
    atomic_numbers,
    masses,
    dataset,
    run=None,
    n_traj=256,
    T_fs=10.0,
    n_ref_steps=512,
    n_steps_list=(1, 2, 4, 8, 16, 32, 64, 128),
    sim_ps=50.0,
    step_sizes=(1.0, 3.0, 5.0, 7.0, 9.0),
    snap_fs=100.0,
    threshold=0.5,
    xlim=10.0,
    n_bins=500,
    seed=0,
    name=None,
    weak_only=False,
):
    base = model.base if isinstance(model, TemperatureCorrectedFlowMap) else model
    if name is not None:
        print(f"\n{name} flow-map evaluation", flush=True)
    if dataset == "aldp":
        _, _, test = load_aldp()
    else:
        _, _, test = load_md17(dataset)
    ref_positions = jnp.asarray(test.positions)
    kT = base.kT
    n_atoms = masses.shape[0]
    poly = BrownianPolynomial(degree=base.network.degree, shape=(n_atoms, 3))
    ref_solver = BAOAB(base)
    em_solver = EulerMaruyama(base)

    key = jr.key(seed)
    key, k_idx, k_vel, k_path, k_roll = jr.split(key, 5)
    idx = jr.choice(k_idx, ref_positions.shape[0], (n_traj,), replace=False)
    x0 = ref_positions[idx]
    v0 = maxwell_boltzmann(k_vel, masses, kT, x0.shape)

    # Strong error over shared Brownian paths, scored against the same fine BAOAB
    # reference roll.
    T = T_fs * units.fs
    I_fine = sample_fine_paths(poly, n_traj, n_ref_steps, T, k_path)
    ref_xs, ref_vs = _reference_trajs(ref_solver, x0, v0, T / n_ref_steps, I_fine)
    fm_results = error_sweep(
        model, poly, x0, v0, T, I_fine, ref_xs, ref_vs, list(n_steps_list)
    )
    em_results = error_sweep(
        em_solver, poly, x0, v0, T, I_fine, ref_xs, ref_vs, list(n_steps_list)
    )
    baoab_results = error_sweep(
        ref_solver, poly, x0, v0, T, I_fine, ref_xs, ref_vs, list(n_steps_list)
    )
    print(
        f"strong / weak error over T = {T_fs:g} fs, {n_traj} traj, "
        f"reference BAOAB at {T_fs / n_ref_steps:.4f} fs:",
        flush=True,
    )
    print(
        f"  {'h/fs':>7}  {'strong fm':>11}  {'strong em':>11}  {'strong ba':>11}  "
        f"{'weak fm x/v':>21}  {'weak ba x/v':>21}",
        flush=True,
    )
    for fm, em, ba in zip(fm_results, em_results, baoab_results):
        print(
            f"  {fm[1] / units.fs:7.3f}  {fm[2]:11.4e}  {em[2]:11.4e}  "
            f"{ba[2]:11.4e}  "
            f"{fm[4]:9.2e} {fm[5]:9.2e}  {ba[4]:9.2e} {ba[5]:9.2e}",
            flush=True,
        )
    fm_ox, fm_ov = weak_orders(fm_results)
    ba_ox, ba_ov = weak_orders(baoab_results)
    print(
        f"  strong order: flow map {strong_order(fm_results):.2f}, "
        f"Euler-Maruyama {strong_order(em_results):.2f}, "
        f"BAOAB {strong_order(baoab_results):.2f}",
        flush=True,
    )
    print(
        f"  weak order:   flow map pos {fm_ox:.2f} vel {fm_ov:.2f}, "
        f"BAOAB pos {ba_ox:.2f} vel {ba_ov:.2f}",
        flush=True,
    )

    if weak_only:
        print(
            f"  {'h/fs':>7}  {'weak fm x +/- SE':>24}  {'weak fm v +/- SE':>24}  "
            f"{'weak ba x +/- SE':>24}  {'weak ba v +/- SE':>24}",
            flush=True,
        )
        for fm, ba in zip(fm_results, baoab_results):
            print(
                f"  {fm[1] / units.fs:7.3f}  {fm[4]:9.2e} +/- {fm[6]:8.2e}  "
                f"{fm[5]:9.2e} +/- {fm[7]:8.2e}  "
                f"{ba[4]:9.2e} +/- {ba[6]:8.2e}  "
                f"{ba[5]:9.2e} +/- {ba[7]:8.2e}",
                flush=True,
            )
            if run is not None:
                n_steps = fm[0]
                suffix = f"_{name}" if name is not None else ""
                run.summary.update(
                    {
                        f"weak_error_fm_pos_{n_steps}{suffix}": fm[4],
                        f"weak_error_fm_vel_{n_steps}{suffix}": fm[5],
                        f"weak_error_fm_pos_se_{n_steps}{suffix}": fm[6],
                        f"weak_error_fm_vel_se_{n_steps}{suffix}": fm[7],
                        f"weak_error_baoab_pos_{n_steps}{suffix}": ba[4],
                        f"weak_error_baoab_vel_{n_steps}{suffix}": ba[5],
                        f"weak_error_baoab_pos_se_{n_steps}{suffix}": ba[6],
                        f"weak_error_baoab_vel_se_{n_steps}{suffix}": ba[7],
                    }
                )
        return

    # One truncated flow-map rollout per step size feeds all distributional metrics.
    bins = jnp.linspace(1e-6, xlim, n_bins + 1)
    bonds = bond_indices(atomic_numbers, x0[0])
    ref_mean = bond_lengths(ref_positions, bonds).mean(axis=0)
    h_ref = hist_density(ref_positions, pairwise_distances, bins)

    print(f"\nflow-map sweep: {n_traj} traj x {sim_ps:g} ps", flush=True)
    t0 = time.time()
    rows = [
        evaluate_rollout(
            model,
            poly,
            x0,
            v0,
            dt_fs,
            sim_ps,
            snap_fs,
            threshold,
            atomic_numbers,
            masses,
            bonds,
            ref_mean,
            bins,
            xlim,
            h_ref,
            kT,
            jr.fold_in(k_roll, int(dt_fs)),
        )
        for dt_fs in step_sizes
    ]
    print(f"  {len(step_sizes)} step sizes in {time.time() - t0:.1f}s", flush=True)

    elements = sorted({int(z) for z in np.asarray(atomic_numbers)})
    T_target = rows[0]["T_target"]

    print_table(
        "h(r) MAE [unitless]",
        ["h/fs", "MAE", "surv/ps", "collapse"],
        [
            [
                f"{r['dt_fs']:g}",
                f"{r['mae']:.4f}",
                f"{r['survival']:.1f}",
                f"{r['collapsed']}/{n_traj}",
            ]
            for r in rows
        ],
    )
    print_table(
        f"kinetic dT [K], target {T_target:.0f} K",
        ["h/fs", "overall"] + [chemical_symbols[z] for z in elements],
        [
            [f"{r['dt_fs']:g}", f"{r['dT']:+.1f}"]
            + [f"{r['dT_el'][z]:+.1f}" for z in elements]  # pyright: ignore
            for r in rows
        ],
    )

    if run is not None:
        fig = plot_summary(
            bins,
            h_ref,
            rows,
            fm_results,
            em_results,
            baoab_results,
            fm_label=name or "flow map",
        )
        suffix = f"_{name}" if name is not None else ""
        run.summary.update(
            {
                f"strong_order_fm{suffix}": strong_order(fm_results),
                f"strong_order_em{suffix}": strong_order(em_results),
                f"strong_order_baoab{suffix}": strong_order(baoab_results),
                f"weak_order_fm_pos{suffix}": fm_ox,
                f"weak_order_fm_vel{suffix}": fm_ov,
                f"weak_order_baoab_pos{suffix}": ba_ox,
                f"weak_order_baoab_vel{suffix}": ba_ov,
            }
        )
        rollout_table = wandb.Table(
            columns=["dt_fs", "h_r_mae", "survival_ps", "collapsed", "dT"]
            + [chemical_symbols[z] for z in elements],  # pyright: ignore
            data=[
                [r["dt_fs"], r["mae"], r["survival"], r["collapsed"], r["dT"]]
                + [r["dT_el"][z] for z in elements]  # pyright: ignore
                for r in rows
            ],
        )
        run.log(
            {
                f"rollout{suffix}": rollout_table,
                f"metrics{suffix}": wandb.Image(fig),
            }
        )


def main(
    run_name=None,
    wandb_project="Langevin Flow Maps",
    output_run_name=None,
    temperature_corrected=False,
    temperature_group_by="element",
    n_traj=256,
    T_fs=10.0,
    n_ref_steps=512,
    n_steps_list=(1, 2, 4, 8, 16, 32, 64, 128),
    sim_ps=50.0,
    step_sizes=(1.0, 3.0, 5.0, 7.0, 9.0),
    snap_fs=100.0,
    threshold=0.5,
    xlim=10.0,
    n_bins=500,
    seed=0,
    weak_only=False,
):
    if run_name is None:
        raise ValueError("run_name is required: the wandb run whose model to evaluate")
    run = wandb.init(
        project=wandb_project,
        name=output_run_name,
        job_type="eval",
        config=dict(
            run_name=run_name,
            temperature_corrected=temperature_corrected,
            temperature_group_by=temperature_group_by,
            n_traj=n_traj,
            T_fs=T_fs,
            n_ref_steps=n_ref_steps,
            sim_ps=sim_ps,
            step_sizes=list(step_sizes),
            snap_fs=snap_fs,
            threshold=threshold,
            seed=seed,
            weak_only=weak_only,
        ),
    )
    artifact_dir = run.use_artifact(f"{run_name}:latest").download()
    model, atomic_numbers, masses, meta = load_model_dir(artifact_dir)
    if temperature_corrected:
        model = TemperatureCorrectedFlowMap.from_model(
            model,
            group_by=temperature_group_by,  # pyright: ignore
        )
    evaluate(
        model,
        atomic_numbers,
        masses,
        meta["dataset"],
        run,
        name="corrected" if temperature_corrected else "original",
        n_traj=n_traj,
        T_fs=T_fs,
        n_ref_steps=n_ref_steps,
        n_steps_list=n_steps_list,
        sim_ps=sim_ps,
        step_sizes=step_sizes,
        snap_fs=snap_fs,
        threshold=threshold,
        xlim=xlim,
        n_bins=n_bins,
        seed=seed,
        weak_only=weak_only,
    )
    run.finish()


if __name__ == "__main__":
    main()
