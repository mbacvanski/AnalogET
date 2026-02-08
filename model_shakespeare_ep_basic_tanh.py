"""Minimal Equilibrium Propagation (EP) training for Shakespeare next-char.

This is a "back to basics" EP implementation with as little extra machinery
as possible, meant for debugging the core algorithm and model dynamics.

Key difference vs the original energy-transformer model:
- Hopfield activation uses tanh instead of ReLU, with matching potential
  L_hopf(h) = sum log cosh(h). This removes the unbounded-below quadratic
  term that ReLU^2 can induce in the reduced energy.

No projection, no gating, no partitioned optimizers by default.
"""

import argparse
import json
import os
import time
from dataclasses import replace
from datetime import datetime

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import jax.random as jr
import optax

try:
    import wandb  # optional
except Exception:  # pragma: no cover
    wandb = None

from config import Config
from data import load_shakespeare_dataset, prepare_shakespeare_dataset
from model import get_xi_attn_embed, get_xi_hopf, init_params, logits_from_v
from utils import ModelParams


# -----------------------------------------------------------------------------
# Position Encoding (positive-used, for analog plausibility)
# -----------------------------------------------------------------------------


def get_xi_pos(params: ModelParams, L: int, D: int) -> jax.Array:
    """(L, D) position embedding, kept nonnegative via squaring.

    If the param is missing, return zeros (for backward compatibility).
    """
    raw = params.get("xi_pos_raw", None)
    if raw is None:
        return jnp.zeros((L, D), dtype=jnp.float32)
    return jnp.square(raw)


# -----------------------------------------------------------------------------
# Energy (tanh Hopfield)
# -----------------------------------------------------------------------------


def L_attn(h: jax.Array, beta: float) -> jax.Array:
    # (B, L) -> (B,)
    return (1.0 / beta) * jax.nn.logsumexp(beta * h, axis=-1)


def L_hopf_tanh(h: jax.Array) -> jax.Array:
    # (B, M) -> (B,)
    # derivative wrt h is tanh(h)
    return jnp.sum(jnp.log(jnp.cosh(h)), axis=-1)


def energy_per_sample_tanh(
    params: ModelParams,
    v: jax.Array,  # (D,)
    h_attn: jax.Array,  # (L,)
    h_hopf: jax.Array,  # (M,)
    f_attn: jax.Array,  # (L,)
    f_hopf: jax.Array,  # (M,)
    ctx_bits_row: jax.Array,  # (L,)
    cfg: Config,
    vis_gain: jax.Array,
) -> jax.Array:
    xi_attn_embed = get_xi_attn_embed(params)  # (vocab, D), positive-used
    xi_pos = get_xi_pos(params, cfg.L, cfg.D)  # (L, D)
    xi_attn = xi_attn_embed[ctx_bits_row] + xi_pos  # (L, D)
    xi_hopf = get_xi_hopf(params)  # (M, D), positive-used

    dv = v - params["a"]
    # Physical "leak" strength; larger keeps equilibrium voltages bounded.
    vis_term = 0.5 * vis_gain * jnp.dot(dv, dv)

    coupling = jnp.dot(v, xi_attn.T @ f_attn + xi_hopf.T @ f_hopf)
    attn_bias = jnp.dot(f_attn, h_attn - params["b"])
    hopf_bias = jnp.dot(f_hopf, h_hopf - params["c"])

    return (
        vis_term
        - coupling
        + attn_bias
        + hopf_bias
        - L_attn(h_attn[None, :], cfg.beta)[0]
        - L_hopf_tanh(h_hopf[None, :])[0]
    )


def energy_per_batch_tanh(
    params: ModelParams,
    V: jax.Array,  # (B, D)
    H_attn: jax.Array,  # (B, L)
    H_hopf: jax.Array,  # (B, M)
    F_attn: jax.Array,  # (B, L)
    F_hopf: jax.Array,  # (B, M)
    ctx_bits: jax.Array,  # (B, L)
    cfg: Config,
    vis_gain: jax.Array,
) -> jax.Array:
    E_b = jax.vmap(
        energy_per_sample_tanh, in_axes=(None, 0, 0, 0, 0, 0, 0, None, None)
    )(params, V, H_attn, H_hopf, F_attn, F_hopf, ctx_bits, cfg, vis_gain)
    return jnp.sum(E_b)


# -----------------------------------------------------------------------------
# Dynamics
# -----------------------------------------------------------------------------


def _init_hidden(params: ModelParams, V0: jax.Array, ctx_bits: jax.Array) -> tuple[jax.Array, jax.Array]:
    xi_attn_embed = get_xi_attn_embed(params)
    xi_pos = get_xi_pos(params, ctx_bits.shape[1], V0.shape[1])  # (L, D)
    batch_xi_attn = xi_attn_embed[ctx_bits] + xi_pos[None, :, :]  # (B, L, D)
    xi_hopf = get_xi_hopf(params)  # (M, D)
    H_attn0 = jnp.einsum("bld,bd->bl", batch_xi_attn, V0) + params["b"]
    H_hopf0 = V0 @ xi_hopf.T + params["c"]
    return H_attn0, H_hopf0


def run_free_phase_tanh(
    params: ModelParams,
    V0: jax.Array,
    ctx_bits: jax.Array,
    cfg: Config,
    vis_gain: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Free-phase inference. Returns (V_T, H_attn_T, H_hopf_T, force_T)."""
    xi_attn_embed = get_xi_attn_embed(params)
    xi_pos = get_xi_pos(params, ctx_bits.shape[1], V0.shape[1])  # (L, D)
    batch_xi_attn = xi_attn_embed[ctx_bits] + xi_pos[None, :, :]  # (B, L, D)
    xi_hopf = get_xi_hopf(params)  # (M, D)

    step_v = cfg.step_size / cfg.tau_v
    step_h = cfg.step_size / cfg.tau_h

    H_attn0, H_hopf0 = _init_hidden(params, V0, ctx_bits)

    def body(_, carry):
        V, H_attn, H_hopf = carry
        F_attn = jax.nn.softmax(cfg.beta * H_attn, axis=-1)
        F_hopf = jnp.tanh(H_hopf)

        dE_dV = (
            (V - params["a"]) * vis_gain
            - jnp.einsum("bld,bl->bd", batch_xi_attn, F_attn)
            - F_hopf @ xi_hopf
        )
        dE_dF_attn = H_attn - (jnp.einsum("bld,bd->bl", batch_xi_attn, V) + params["b"])
        dE_dF_hopf = H_hopf - (V @ xi_hopf.T + params["c"])

        V = V - step_v * dE_dV
        H_attn = H_attn - step_h * dE_dF_attn
        H_hopf = H_hopf - step_h * dE_dF_hopf
        return (V, H_attn, H_hopf)

    V_T, H_attn_T, H_hopf_T = jax.lax.fori_loop(0, cfg.n_steps, body, (V0, H_attn0, H_hopf0))

    # terminal force for monitoring (dV/dt)
    F_attn_T = jax.nn.softmax(cfg.beta * H_attn_T, axis=-1)
    F_hopf_T = jnp.tanh(H_hopf_T)
    dE_dV_T = (
        (V_T - params["a"]) * vis_gain
        - jnp.einsum("bld,bl->bd", batch_xi_attn, F_attn_T)
        - F_hopf_T @ xi_hopf
    )
    force_T = -(1.0 / cfg.tau_v) * dE_dV_T
    return V_T, H_attn_T, H_hopf_T, force_T


def run_nudged_phase_tanh(
    params: ModelParams,
    V_init: jax.Array,
    H_attn_init: jax.Array,
    H_hopf_init: jax.Array,
    ctx_bits: jax.Array,
    labels: jax.Array,
    beta_nudge: jax.Array,
    cfg: Config,
    n_nudge_steps: int,
    vis_gain: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Nudged-phase inference. Returns (V_T, H_attn_T, H_hopf_T, force_T)."""
    xi_attn_embed = get_xi_attn_embed(params)
    xi_pos = get_xi_pos(params, ctx_bits.shape[1], V_init.shape[1])  # (L, D)
    batch_xi_attn = xi_attn_embed[ctx_bits] + xi_pos[None, :, :]
    xi_hopf = get_xi_hopf(params)

    step_v = cfg.step_size / cfg.tau_v
    step_h = cfg.step_size / cfg.tau_h

    def body(_, carry):
        V, H_attn, H_hopf = carry
        F_attn = jax.nn.softmax(cfg.beta * H_attn, axis=-1)
        F_hopf = jnp.tanh(H_hopf)

        dE_dV = (
            (V - params["a"]) * vis_gain
            - jnp.einsum("bld,bl->bd", batch_xi_attn, F_attn)
            - F_hopf @ xi_hopf
        )
        dE_dF_attn = H_attn - (jnp.einsum("bld,bd->bl", batch_xi_attn, V) + params["b"])
        dE_dF_hopf = H_hopf - (V @ xi_hopf.T + params["c"])

        logits = logits_from_v(params, V)
        probs = jax.nn.softmax(logits, axis=-1)
        one_hot_y = jax.nn.one_hot(labels, cfg.vocab_size)
        dC_dV = (probs - one_hot_y) @ params["W_dec"] / V.shape[0]

        V = V - step_v * (dE_dV + beta_nudge * dC_dV)
        H_attn = H_attn - step_h * dE_dF_attn
        H_hopf = H_hopf - step_h * dE_dF_hopf
        return (V, H_attn, H_hopf)

    V_T, H_attn_T, H_hopf_T = jax.lax.fori_loop(
        0, n_nudge_steps, body, (V_init, H_attn_init, H_hopf_init)
    )

    # terminal force for monitoring (dV/dt of the nudged dynamics)
    F_attn_T = jax.nn.softmax(cfg.beta * H_attn_T, axis=-1)
    F_hopf_T = jnp.tanh(H_hopf_T)
    dE_dV_T = (
        (V_T - params["a"]) * vis_gain
        - jnp.einsum("bld,bl->bd", batch_xi_attn, F_attn_T)
        - F_hopf_T @ xi_hopf
    )
    logits_T = logits_from_v(params, V_T)
    probs_T = jax.nn.softmax(logits_T, axis=-1)
    one_hot_y = jax.nn.one_hot(labels, cfg.vocab_size)
    dC_dV_T = (probs_T - one_hot_y) @ params["W_dec"] / V_T.shape[0]
    force_T = -(1.0 / cfg.tau_v) * (dE_dV_T + beta_nudge * dC_dV_T)
    return V_T, H_attn_T, H_hopf_T, force_T


# -----------------------------------------------------------------------------
# EP step
# -----------------------------------------------------------------------------


def ce_loss_from_v(params: ModelParams, V: jax.Array, y: jax.Array) -> jax.Array:
    logits = logits_from_v(params, V)
    return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, y))


def main():
    parser = argparse.ArgumentParser(description="Basic EP (tanh Hopfield) on Shakespeare")
    parser.add_argument("--config", type=str, default="shakespeare_config.json")
    parser.add_argument("--max_steps", type=int, default=None, help="Override config max_steps")
    parser.add_argument("--lr", type=float, default=None, help="Override config lr_peak_value")
    parser.add_argument(
        "--attn_beta",
        type=float,
        default=None,
        help="Override attention inverse temperature beta (higher => sharper attention).",
    )
    parser.add_argument("--beta_nudge", type=float, default=0.5)
    parser.add_argument("--nudge_frac", type=float, default=0.5)
    parser.add_argument(
        "--xi_pos_raw_scale",
        type=float,
        default=0.1,
        help="Init scale for positive-used position embedding (xi_pos = square(raw)).",
    )
    parser.add_argument(
        "--vis_gain",
        type=float,
        default=1.0,
        help="Multiplier on the visible quadratic term 0.5*vis_gain*||V-a||^2 (stronger leak keeps V bounded).",
    )
    parser.add_argument("--sample_mode", action="store_true")
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=None,
        help="AdamW weight decay (overrides config slow_weight_decay).",
    )
    args = parser.parse_args()

    # ---- Data ----
    filename_prefix = "shakespeare_data"
    dataset_exists = os.path.exists(f"data/{filename_prefix}_train_X.txt")
    if dataset_exists:
        train_X, train_y, valid_X, valid_y, char_to_idx, idx_to_char = load_shakespeare_dataset(
            filename_prefix=filename_prefix
        )
    else:
        train_X, train_y, valid_X, valid_y, char_to_idx, idx_to_char = prepare_shakespeare_dataset(
            ctx_length=16, train_ratio=0.9, filename_prefix=filename_prefix
        )
    vocab_size = len(char_to_idx)

    if args.sample_mode:
        max_samples = 1000 * 256
        train_X = train_X[:max_samples]
        train_y = train_y[:max_samples]
        valid_X = valid_X[:max_samples]
        valid_y = valid_y[:max_samples]

    # ---- Config ----
    with open(args.config, "r") as f:
        d = json.load(f)
    cfg = Config(
        L=d.get("L", 16),
        vocab_size=vocab_size,
        **{k: v for k, v in d.items() if k not in ("L", "vocab_size")},
    )
    if args.max_steps is not None:
        cfg = replace(cfg, max_steps=int(args.max_steps))
    if args.lr is not None:
        cfg = replace(cfg, lr_peak_value=float(args.lr))
    if args.attn_beta is not None:
        cfg = replace(cfg, beta=float(args.attn_beta))

    n_nudge_steps = max(1, int(cfg.n_steps * args.nudge_frac))
    beta_val = jnp.asarray(args.beta_nudge, dtype=jnp.float32)
    vis_gain = jnp.asarray(args.vis_gain, dtype=jnp.float32)

    # ---- Init ----
    key = jr.PRNGKey(cfg.seed)
    params = init_params(key, cfg)
    # Add a positive-used position encoding for attention rows: xi_attn[pos] = token_embed[token] + xi_pos[pos]
    key, kpos = jr.split(key)
    params["xi_pos_raw"] = jr.normal(kpos, (cfg.L, cfg.D)) * float(args.xi_pos_raw_scale)

    # ---- Optimizer (single AdamW) ----
    lr_sched = optax.warmup_cosine_decay_schedule(
        init_value=cfg.lr_init_value,
        peak_value=cfg.lr_peak_value,
        warmup_steps=min(100, max(1, cfg.max_steps // 10)),
        decay_steps=max(1, cfg.max_steps - min(100, max(1, cfg.max_steps // 10))),
        end_value=cfg.lr_peak_value * cfg.lr_end_factor,
    )
    wd = cfg.slow_weight_decay if args.weight_decay is None else float(args.weight_decay)
    tx = optax.chain(
        optax.clip_by_global_norm(cfg.max_norm),
        optax.adamw(lr_sched, weight_decay=wd),
    )
    opt_state = tx.init(params)

    # ---- EP forward (jit) ----
    @jax.jit
    def ep_grads_and_metrics(p, bx, by):
        B = bx.shape[0]
        V0 = jnp.zeros((B, cfg.D), dtype=jnp.float32)

        V_free, H_attn_free, H_hopf_free, force_free = run_free_phase_tanh(
            p, V0, bx, cfg, vis_gain
        )
        # Light diagnostics about whether the fixed point is doing anything "attention-like".
        F_attn_free = jax.nn.softmax(cfg.beta * H_attn_free, axis=-1)
        F_hopf_free = jnp.tanh(H_hopf_free)
        attn_entropy = -jnp.sum(F_attn_free * jnp.log(jnp.maximum(F_attn_free, 1e-20)), axis=-1)
        attn_entropy = jnp.mean(attn_entropy)  # nats
        attn_max = jnp.mean(jnp.max(F_attn_free, axis=-1))
        hopf_sat = jnp.mean(jnp.abs(F_hopf_free))
        V_free = jax.lax.stop_gradient(V_free)
        H_attn_free = jax.lax.stop_gradient(H_attn_free)
        H_hopf_free = jax.lax.stop_gradient(H_hopf_free)

        V_n, H_attn_n, H_hopf_n, force_n = run_nudged_phase_tanh(
            p,
            V_free,
            H_attn_free,
            H_hopf_free,
            bx,
            by,
            beta_val,
            cfg,
            n_nudge_steps,
            vis_gain,
        )
        V_n = jax.lax.stop_gradient(V_n)
        H_attn_n = jax.lax.stop_gradient(H_attn_n)
        H_hopf_n = jax.lax.stop_gradient(H_hopf_n)

        def E_params(pp, V, H_attn, H_hopf):
            F_attn = jax.nn.softmax(cfg.beta * H_attn, axis=-1)
            F_hopf = jnp.tanh(H_hopf)
            return energy_per_batch_tanh(
                pp, V, H_attn, H_hopf, F_attn, F_hopf, bx, cfg, vis_gain
            )

        dE_free = jax.grad(E_params)(p, V_free, H_attn_free, H_hopf_free)
        dE_n = jax.grad(E_params)(p, V_n, H_attn_n, H_hopf_n)
        ep_g = jax.tree.map(lambda n, f: (n - f) / beta_val, dE_n, dE_free)

        def dec_cost(pp):
            return ce_loss_from_v(pp, V_free, by)

        dec_g = jax.grad(dec_cost)(p)
        g = jax.tree.map(jnp.add, ep_g, dec_g)

        loss = ce_loss_from_v(p, V_free, by)
        force_rms = jnp.sqrt(jnp.mean(jnp.sum(force_free * force_free, axis=1)))
        disp_rms = jnp.sqrt(jnp.mean(jnp.sum((V_n - V_free) ** 2, axis=1)))
        v_rms = jnp.sqrt(jnp.mean(jnp.sum(V_free * V_free, axis=1)))

        metrics = dict(
            loss=loss,
            force_rms=force_rms,
            disp_rms=disp_rms,
            v_rms=v_rms,
            force_rel=force_rms / jnp.maximum(v_rms, 1e-8),
            disp_rel=disp_rms / jnp.maximum(v_rms, 1e-8),
            attn_entropy=attn_entropy,
            attn_max=attn_max,
            hopf_sat=hopf_sat,
        )
        return g, metrics

    @jax.jit
    def train_step(p, s, bx, by):
        g, m = ep_grads_and_metrics(p, bx, by)
        updates, s2 = tx.update(g, s, p)
        p2 = optax.apply_updates(p, updates)
        upd_norm = optax.global_norm(updates)
        grad_norm = optax.global_norm(g)
        m = dict(m)
        m["grad_norm"] = grad_norm
        m["update_norm"] = upd_norm
        return p2, s2, m

    # ---- Logging setup ----
    run = None
    if args.wandb:
        if wandb is None:
            raise RuntimeError("wandb is not installed but --wandb was set.")
        run = wandb.init(
            entity="qpaig",
            project="analog-et",
            name=f"EP_basic_tanh_ctx{cfg.L}_bs{cfg.batch_size}",
            config={
                "script": "model_shakespeare_ep_basic_tanh.py",
                "beta_nudge": float(args.beta_nudge),
                "nudge_frac": float(args.nudge_frac),
                "n_nudge_steps": int(n_nudge_steps),
                "vis_gain": float(args.vis_gain),
                "xi_pos_raw_scale": float(args.xi_pos_raw_scale),
                "config_file": args.config,
                "sample_mode": bool(args.sample_mode),
                "lr_peak": float(cfg.lr_peak_value),
                "weight_decay": float(wd),
                "max_steps": int(cfg.max_steps),
            },
        )

    # ---- Train loop ----
    num_train = train_X.shape[0]
    num_batches = (num_train + cfg.batch_size - 1) // cfg.batch_size

    t0 = time.time()
    global_step = 0
    for epoch in range(10_000_000):  # break by max_steps
        key, kperm = jr.split(key)
        perm = jr.permutation(kperm, num_train)
        Xep = train_X[perm]
        yep = train_y[perm]

        for b in range(num_batches):
            if global_step >= cfg.max_steps:
                break
            start = b * cfg.batch_size
            stop = min(start + cfg.batch_size, num_train)
            bx = Xep[start:stop]
            by = yep[start:stop]
            if stop <= start:
                continue

            params, opt_state, m = train_step(params, opt_state, bx, by)

            if global_step % args.log_every == 0:
                dt = time.time() - t0
                print(
                    f"step {global_step:6d}  "
                    f"loss {float(m['loss']):.4f}  "
                    f"v_rms {float(m['v_rms']):.3e}  "
                    f"force_rel {float(m['force_rel']):.3e}  "
                    f"disp_rel {float(m['disp_rel']):.3e}  "
                    f"grad {float(m['grad_norm']):.3e}  "
                    f"upd {float(m['update_norm']):.3e}  "
                    f"sec {dt:.1f}"
                )
                t0 = time.time()

            if run is not None:
                wandb.log(
                    {
                        "train/loss": float(m["loss"]),
                        "train/force_rel": float(m["force_rel"]),
                        "train/disp_rel": float(m["disp_rel"]),
                        "train/v_rms": float(m["v_rms"]),
                        "train/force_rms": float(m["force_rms"]),
                        "train/disp_rms": float(m["disp_rms"]),
                        "train/attn_entropy": float(m["attn_entropy"]),
                        "train/attn_max": float(m["attn_max"]),
                        "train/hopf_sat": float(m["hopf_sat"]),
                        "train/grad_norm": float(m["grad_norm"]),
                        "train/update_norm": float(m["update_norm"]),
                        "train/lr": float(lr_sched(global_step)),
                    },
                    step=global_step,
                )

            if global_step % args.eval_every == 0 and global_step > 0:
                # quick eval on a single batch
                vx = valid_X[: cfg.batch_size]
                vy = valid_y[: cfg.batch_size]
                gtmp, mt = ep_grads_and_metrics(params, vx, vy)
                if run is not None:
                    wandb.log({"valid/loss": float(mt["loss"])}, step=global_step)
                print(f"  valid(loss) {float(mt['loss']):.4f}")

            global_step += 1

        if global_step >= cfg.max_steps:
            break

    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
