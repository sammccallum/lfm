import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr


def _atom_mask(atom_mask, n_atoms):
    if atom_mask is None:
        return jnp.ones((n_atoms,), dtype=bool)
    return jnp.asarray(atom_mask, dtype=bool)


def cutoff_softmax(scores, cutoff):
    edge_mask = cutoff > 0.0
    log_cutoff = jnp.log(jnp.where(edge_mask, cutoff, 1.0))
    logits = jnp.where(edge_mask[:, :, None], scores + log_cutoff[:, :, None], -jnp.inf)
    maximum = jnp.max(logits, axis=1, keepdims=True)
    maximum = jnp.where(jnp.isfinite(maximum), maximum, 0.0)
    weights = jnp.exp(logits - jax.lax.stop_gradient(maximum))
    normalizer = jnp.sum(weights, axis=1, keepdims=True)
    return weights / jnp.where(normalizer > 0.0, normalizer, 1.0)


class RelativePositionalEmbedding(eqx.Module):
    mlp: eqx.nn.MLP
    num_basis: int
    cutoff: float

    def __init__(self, hidden_size, key, num_basis=10, cutoff=7.5):
        self.num_basis = num_basis
        self.cutoff = cutoff
        self.mlp = eqx.nn.MLP(
            num_basis + 3,
            hidden_size,
            hidden_size,
            1,
            activation=jax.nn.silu,
            key=key,
        )

    def _basic_fourier(self, d):
        freqs = jnp.pi * jnp.arange(self.num_basis)
        return jnp.cos(freqs * (d[..., None] / self.cutoff))

    def _smooth_cutoff(self, d):
        # e3x smooth_cutoff: exp(1 - 1/(1 - (d/c)^2)) for d < c, else 0.
        x = d / self.cutoff
        inside = x < 1.0
        safe = jnp.where(inside, x, 0.0)

        return jnp.where(inside, jnp.exp(1.0 - 1.0 / (1.0 - safe * safe)), 0.0)

    def __call__(self, pos, atom_mask=None):
        # pos: (N, 3) -> edge embedding (N, N, hidden_size), cutoff (N, N).
        disp = pos[:, None, :] - pos[None, :, :]
        d = jnp.sqrt(jnp.sum(disp**2, axis=-1) + 1e-12)
        rhat = disp / (d[..., None] + 1e-9)
        rbf = self._basic_fourier(d)

        e_in = jnp.concatenate([rbf, rhat], axis=-1)
        e = jax.vmap(jax.vmap(self.mlp))(e_in)
        atom_mask = _atom_mask(atom_mask, pos.shape[0])
        pair_mask = atom_mask[:, None] & atom_mask[None, :]

        return (
            jnp.where(pair_mask[..., None], e, 0.0),
            jnp.where(pair_mask, self._smooth_cutoff(d), 0.0),
        )


class AdaptiveLayerNorm(eqx.Module):
    """Layer norm with conditioning-driven scale and shift: LN(h) * (1 + gamma) + beta."""

    norm: eqx.nn.LayerNorm

    def __init__(self, hidden_size):
        self.norm = eqx.nn.LayerNorm(hidden_size, use_weight=False, use_bias=False)

    def __call__(self, h, gamma, beta):
        # h, gamma, beta: (H,)
        return self.norm(h) * (1.0 + gamma) + beta


class AdaptiveScale(eqx.Module):
    """Conditioning-driven gating: h * alpha."""

    def __call__(self, h, alpha):
        # h, alpha: (H,)
        return h * alpha


class GaussianFourierFeatures(eqx.Module):
    """Random Fourier features with a learnable frequency projection."""

    b: jax.Array

    def __init__(self, in_size, num_features, key, sigma=1.0):
        assert num_features % 2 == 0
        self.b = sigma * jr.normal(key, (in_size, num_features // 2))

    def __call__(self, x):
        # x: (in_size,) -> (num_features,)
        proj = 2.0 * jnp.pi * (x @ self.b)
        return jnp.concatenate([jnp.cos(proj), jnp.sin(proj)])


def _soft_gaussian(d, num_basis, d_max):
    # d: scalar -> (num_basis,); Gaussian RBFs on [0, d_max].
    values = jnp.linspace(0.0, d_max, num_basis + 2)
    step = values[1] - values[0]
    centers = values[1:-1]
    diff = (d - centers) / step
    return (num_basis**0.5) * jnp.exp(-(diff**2)) / 1.12


class Conditioning(eqx.Module):
    """Per-atom conditioning token c from time, velocity and mass."""

    time_ff: GaussianFourierFeatures
    time_mlp: eqx.nn.MLP
    vel_mix: eqx.nn.Embedding
    vel_mlp: eqx.nn.MLP
    mass_mlp: eqx.nn.MLP
    fuse_mlp: eqx.nn.MLP
    num_vel_basis: int
    vel_max: float

    def __init__(
        self,
        hidden_size,
        key,
        n_species=100,
        num_time_fourier=None,
        num_vel_basis=8,
        vel_max=1.55,
    ):
        num_time_fourier = (
            hidden_size // 2 if num_time_fourier is None else num_time_fourier
        )
        kt, ktm, kv, kvm, kmm, kf = jr.split(key, 6)
        self.time_ff = GaussianFourierFeatures(1, num_time_fourier, kt)
        self.time_mlp = eqx.nn.MLP(
            num_time_fourier,
            hidden_size,
            hidden_size,
            1,
            activation=jax.nn.silu,
            key=ktm,
        )
        # Atom-specific mixing W_z of the velocity radial basis (one-hot @ W == lookup).
        self.vel_mix = eqx.nn.Embedding(
            n_species, num_vel_basis * num_vel_basis, key=kv
        )
        self.vel_mlp = eqx.nn.MLP(
            num_vel_basis + 3,
            hidden_size,
            hidden_size,
            1,
            activation=jax.nn.silu,
            key=kvm,
        )
        self.mass_mlp = eqx.nn.MLP(
            1, hidden_size, hidden_size, 1, activation=jax.nn.silu, key=kmm
        )
        self.fuse_mlp = eqx.nn.MLP(
            3 * hidden_size, hidden_size, hidden_size, 1, activation=jax.nn.silu, key=kf
        )
        self.num_vel_basis = num_vel_basis
        self.vel_max = vel_max

    def __call__(self, dt, velocity, mass, atomic_number):
        # dt, mass: scalar, velocity: (3,), atomic_number: int scalar -> c: (H,)
        c_t = self.time_mlp(self.time_ff(dt[None]))

        v_norm = jnp.sqrt(jnp.sum(velocity**2) + 1e-12)
        rbf = _soft_gaussian(v_norm, self.num_vel_basis, self.vel_max)
        W = self.vel_mix(atomic_number).reshape(self.num_vel_basis, self.num_vel_basis)
        rbf = rbf @ W
        c_v = self.vel_mlp(jnp.concatenate([rbf, velocity]))

        c_m = self.mass_mlp(mass[None])

        return self.fuse_mlp(jnp.concatenate([c_t, c_v, c_m]))


class LFMConditioning(eqx.Module):
    """Per-atom conditioning token c from time, velocity, mass and the Brownian
    coefficients I_st. The I_st branch mirrors the velocity branch per Legendre
    coefficient, adding a fourth fused token c_I."""

    time_ff: GaussianFourierFeatures
    time_mlp: eqx.nn.MLP
    vel_mix: eqx.nn.Embedding
    vel_mlp: eqx.nn.MLP
    noise_mlp: eqx.nn.MLP
    mass_mlp: eqx.nn.MLP
    fuse_mlp: eqx.nn.MLP
    num_vel_basis: int
    vel_max: float
    num_noise_basis: int
    noise_max: float
    degree: int

    def __init__(
        self,
        hidden_size,
        key,
        n_species=100,
        num_time_fourier=None,
        num_vel_basis=8,
        vel_max=1.55,
        degree=3,
        num_noise_basis=8,
        noise_max=0.75,
    ):
        num_time_fourier = (
            hidden_size // 2 if num_time_fourier is None else num_time_fourier
        )
        kt, ktm, kv, kvm, kmm, kf, knmlp = jr.split(key, 7)
        self.time_ff = GaussianFourierFeatures(1, num_time_fourier, kt)
        self.time_mlp = eqx.nn.MLP(
            num_time_fourier,
            hidden_size,
            hidden_size,
            1,
            activation=jax.nn.silu,
            key=ktm,
        )
        self.vel_mix = eqx.nn.Embedding(
            n_species, num_vel_basis * num_vel_basis, key=kv
        )
        self.vel_mlp = eqx.nn.MLP(
            num_vel_basis + 3,
            hidden_size,
            hidden_size,
            1,
            activation=jax.nn.silu,
            key=kvm,
        )
        # Featurise each of the `degree` coefficients like a velocity, then fuse.
        self.noise_mlp = eqx.nn.MLP(
            degree * (num_noise_basis + 3),
            hidden_size,
            hidden_size,
            1,
            activation=jax.nn.silu,
            key=knmlp,
        )
        self.mass_mlp = eqx.nn.MLP(
            1, hidden_size, hidden_size, 1, activation=jax.nn.silu, key=kmm
        )
        self.fuse_mlp = eqx.nn.MLP(
            4 * hidden_size, hidden_size, hidden_size, 1, activation=jax.nn.silu, key=kf
        )
        self.num_vel_basis = num_vel_basis
        self.vel_max = vel_max
        self.num_noise_basis = num_noise_basis
        self.noise_max = noise_max
        self.degree = degree

    def _c_noise(self, I_st):
        # I_st: (degree, 3) -> (H,)
        def per_coeff(I_n):
            norm = jnp.sqrt(jnp.sum(I_n**2) + 1e-12)
            rbf = _soft_gaussian(norm, self.num_noise_basis, self.noise_max)
            return jnp.concatenate([rbf, I_n])

        feats = jax.vmap(per_coeff)(I_st)  # (degree, num_noise_basis + 3)
        return self.noise_mlp(feats.reshape(-1))

    def __call__(self, dt, velocity, mass, atomic_number, I_st):
        # dt, mass: scalar, velocity: (3,), atomic_number: int scalar,
        # I_st: (degree, 3) -> c: (H,)
        c_t = self.time_mlp(self.time_ff(dt[None]))

        v_norm = jnp.sqrt(jnp.sum(velocity**2) + 1e-12)
        rbf = _soft_gaussian(v_norm, self.num_vel_basis, self.vel_max)
        W = self.vel_mix(atomic_number).reshape(self.num_vel_basis, self.num_vel_basis)
        rbf = rbf @ W
        c_v = self.vel_mlp(jnp.concatenate([rbf, velocity]))

        c_I = self._c_noise(I_st)
        c_m = self.mass_mlp(mass[None])

        return self.fuse_mlp(jnp.concatenate([c_t, c_v, c_I, c_m]))


class SelfAttention(eqx.Module):
    W_q: eqx.nn.Linear
    W_k: eqx.nn.Linear
    W_ke: eqx.nn.Linear
    W_v: eqx.nn.Linear
    W_ve: eqx.nn.Linear
    W_o: eqx.nn.Linear
    hidden_size: int
    n_heads: int
    head_dim: int

    def _init_linear(self, key):
        return eqx.nn.Linear(
            self.hidden_size, self.hidden_size, use_bias=False, key=key
        )

    def __init__(self, hidden_size, n_heads, key):
        assert hidden_size % n_heads == 0
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.head_dim = hidden_size // n_heads

        keys = jr.split(key, 6)
        self.W_q = self._init_linear(keys[0])
        self.W_k = self._init_linear(keys[1])
        self.W_ke = self._init_linear(keys[2])
        self.W_v = self._init_linear(keys[3])
        self.W_ve = self._init_linear(keys[4])
        self.W_o = eqx.nn.Linear(hidden_size, hidden_size, use_bias=True, key=keys[5])

    def _split_heads(self, x):
        # (..., H) -> (..., n_heads, head_dim)
        return x.reshape(*x.shape[:-1], self.n_heads, self.head_dim)

    def __call__(self, h, pos_emb, cutoff):
        # h: (N, H), pos_emb: (N, N, H), cutoff: (N, N)
        # a=n_heads, d=head_dim

        N = h.shape[0]

        Q = self._split_heads(jax.vmap(self.W_q)(h))  # (N, a, d)
        K = self._split_heads(
            jax.vmap(self.W_k)(h)[None, :, :] * jax.vmap(jax.vmap(self.W_ke))(pos_emb)
        )  # (N, N, a, d)
        V = self._split_heads(
            jax.vmap(self.W_v)(h)[None, :, :] * jax.vmap(jax.vmap(self.W_ve))(pos_emb)
        )  # (N, N, a, d)

        scores = jnp.einsum("iad,ijad->ija", Q, K) / jnp.sqrt(self.head_dim)
        # Incorporate the cutoff before max subtraction for stable backward.
        edge_mask = cutoff > 0.0
        alpha = cutoff_softmax(scores, cutoff)
        out = jnp.einsum("ija,ijad->iad", alpha, V)  # (N, a, d)
        out = out.reshape(N, self.hidden_size)  # concat heads
        # Return the attention delta only; the residual is owned by TransformerBlock
        # so it stays on the raw (un-normalised) token stream.
        query_mask = jnp.any(edge_mask, axis=1)
        return jax.vmap(self.W_o)(out) * query_mask[:, None]


class TransformerBlock(eqx.Module):
    norm1: eqx.nn.LayerNorm
    norm2: eqx.nn.LayerNorm
    attention: SelfAttention
    mlp: eqx.nn.MLP

    def __init__(self, hidden_size, n_heads, key, ff_mult=4):
        ka, km = jr.split(key, 2)
        self.norm1 = eqx.nn.LayerNorm(hidden_size)
        self.norm2 = eqx.nn.LayerNorm(hidden_size)
        self.attention = SelfAttention(hidden_size, n_heads, ka)
        # 2-layer feed-forward (Linear -> SiLU -> Linear) with an ff_mult-wide hidden.
        self.mlp = eqx.nn.MLP(
            hidden_size,
            hidden_size,
            ff_mult * hidden_size,
            1,
            activation=jax.nn.silu,
            key=km,
        )

    def __call__(self, tokens, pos_emb, cutoff):
        # tokens: (N, H), pos_emb: (N, N, H), cutoff: (N, N)
        h = tokens + self.attention(jax.vmap(self.norm1)(tokens), pos_emb, cutoff)
        h = h + jax.vmap(self.mlp)(jax.vmap(self.norm2)(h))
        return h


def _zero_linear(in_size, out_size, key):
    lin = eqx.nn.Linear(in_size, out_size, key=key)
    bias = lin.bias
    assert bias is not None
    return eqx.tree_at(
        lambda m: (m.weight, m.bias),
        lin,
        (jnp.zeros_like(lin.weight), jnp.zeros_like(bias)),
    )


class AdaptiveTransformerBlock(eqx.Module):
    cond_norm: eqx.nn.LayerNorm
    proj: eqx.nn.Linear
    adaln1: AdaptiveLayerNorm
    adaln2: AdaptiveLayerNorm
    scale: AdaptiveScale
    attention: SelfAttention
    mlp: eqx.nn.MLP

    def __init__(self, hidden_size, n_heads, key, ff_mult=4):
        kp, ka, km = jr.split(key, 3)
        self.cond_norm = eqx.nn.LayerNorm(hidden_size)
        self.proj = _zero_linear(hidden_size, 6 * hidden_size, kp)
        self.adaln1 = AdaptiveLayerNorm(hidden_size)
        self.adaln2 = AdaptiveLayerNorm(hidden_size)
        self.scale = AdaptiveScale()
        self.attention = SelfAttention(hidden_size, n_heads, ka)
        self.mlp = eqx.nn.MLP(
            hidden_size,
            hidden_size,
            ff_mult * hidden_size,
            1,
            activation=jax.nn.gelu,
            key=km,
        )

    def __call__(self, tokens, pos_emb, cutoff, c, atom_mask=None):
        # tokens: (N, H), pos_emb: (N, N, H), cutoff: (N, N), c: (N, H)
        g1, b1, a1, g2, b2, a2 = jnp.split(
            jax.vmap(self.proj)(jax.nn.silu(jax.vmap(self.cond_norm)(c))), 6, axis=-1
        )

        h = jax.vmap(self.adaln1)(tokens, g1, b1)
        tokens = tokens + jax.vmap(self.scale)(self.attention(h, pos_emb, cutoff), a1)

        h = jax.vmap(self.adaln2)(tokens, g2, b2)
        tokens = tokens + jax.vmap(self.scale)(jax.vmap(self.mlp)(h), a2)
        return tokens * _atom_mask(atom_mask, tokens.shape[0])[:, None]


class Readout(eqx.Module):
    cond_norm: eqx.nn.LayerNorm
    proj: eqx.nn.Linear
    adaln_v: AdaptiveLayerNorm
    adaln_f: AdaptiveLayerNorm
    adaln_e: AdaptiveLayerNorm
    head_v: eqx.nn.MLP
    head_f: eqx.nn.MLP
    head_e: eqx.nn.MLP

    def __init__(self, hidden_size, key):
        kp, kv, kf, ke = jr.split(key, 4)
        self.cond_norm = eqx.nn.LayerNorm(hidden_size)
        self.proj = _zero_linear(hidden_size, 6 * hidden_size, kp)
        self.adaln_v = AdaptiveLayerNorm(hidden_size)
        self.adaln_f = AdaptiveLayerNorm(hidden_size)
        self.adaln_e = AdaptiveLayerNorm(hidden_size)
        self.head_v = eqx.nn.MLP(
            hidden_size, 3, hidden_size, 1, activation=jax.nn.silu, key=kv
        )
        self.head_f = eqx.nn.MLP(
            hidden_size, 3, hidden_size, 1, activation=jax.nn.silu, key=kf
        )
        self.head_e = eqx.nn.MLP(
            hidden_size, 1, hidden_size, 1, activation=jax.nn.silu, key=ke
        )

    def __call__(self, tokens, c, atom_mask=None):
        # tokens: (N, H), c: (N, H) -> mean_v (N, 3), mean_f (N, 3), energy scalar
        gv, bv, gf, bf, ge, be = jnp.split(
            jax.vmap(self.proj)(jax.nn.silu(jax.vmap(self.cond_norm)(c))), 6, axis=-1
        )

        mean_v = jax.vmap(self.head_v)(jax.vmap(self.adaln_v)(tokens, gv, bv))
        mean_f = jax.vmap(self.head_f)(jax.vmap(self.adaln_f)(tokens, gf, bf))
        atom_e = jax.vmap(self.head_e)(jax.vmap(self.adaln_e)(tokens, ge, be))
        atom_mask = _atom_mask(atom_mask, tokens.shape[0])
        return (
            mean_v * atom_mask[:, None],
            mean_f * atom_mask[:, None],
            jnp.sum(atom_e[:, 0] * atom_mask),
        )


class LFMTransformer(eqx.Module):
    embed: eqx.nn.Embedding
    pos_embed: RelativePositionalEmbedding
    cond: LFMConditioning
    blocks: list
    readout: Readout
    degree: int

    def __init__(
        self,
        hidden_size,
        n_heads,
        n_blocks,
        key,
        n_species=100,
        num_basis=10,
        cutoff=7.5,
        ff_mult=4,
        num_vel_basis=8,
        vel_max=1.55,
        degree=3,
        num_noise_basis=8,
        noise_max=0.75,
    ):
        ke, kp, kc, kr, *kb = jr.split(key, 4 + n_blocks)
        self.embed = eqx.nn.Embedding(n_species, hidden_size, key=ke)
        self.pos_embed = RelativePositionalEmbedding(
            hidden_size, kp, num_basis=num_basis, cutoff=cutoff
        )
        self.cond = LFMConditioning(
            hidden_size,
            kc,
            n_species=n_species,
            num_vel_basis=num_vel_basis,
            vel_max=vel_max,
            degree=degree,
            num_noise_basis=num_noise_basis,
            noise_max=noise_max,
        )
        self.blocks = [
            AdaptiveTransformerBlock(hidden_size, n_heads, k, ff_mult=ff_mult)
            for k in kb
        ]
        self.readout = Readout(hidden_size, kr)
        self.degree = degree

    def __call__(
        self,
        positions,
        atomic_numbers,
        velocities,
        masses,
        dt,
        I_st,
        atom_mask=None,
    ):
        # positions/velocities: (N, 3), atomic_numbers/masses: (N,), dt: scalar,
        # I_st: (degree, N, 3). Returns mean_v (N, 3), mean_f (N, 3), energy scalar.
        N = positions.shape[0]
        atom_mask = _atom_mask(atom_mask, N)
        tokens = jax.vmap(self.embed)(atomic_numbers) * atom_mask[:, None]
        pos_emb, cutoff = self.pos_embed(positions, atom_mask)
        dt_atoms = jnp.broadcast_to(dt, (N,))
        I_atoms = jnp.swapaxes(I_st, 0, 1)  # (N, degree, 3)
        c = jax.vmap(self.cond)(
            dt_atoms, velocities, masses, atomic_numbers, I_atoms
        )  # (N, H)
        c = c * atom_mask[:, None]

        for block in self.blocks:
            tokens = block(tokens, pos_emb, cutoff, c, atom_mask)

        return self.readout(tokens, c, atom_mask)
