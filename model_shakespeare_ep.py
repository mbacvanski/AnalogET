"""Equilibrium Propagation training for the Shakespeare character prediction model.

Instead of backpropagating through the unrolled ODE dynamics (BPTT),
this uses Equilibrium Propagation (Scellier & Bengio, 2017):

1. Free phase: run energy dynamics to equilibrium
2. Nudged phase: run dynamics with output cost added to energy
3. Parameter gradients = (1/beta) * (dE/dtheta|nudged - dE/dtheta|free)
   + standard decoder gradients evaluated at the free-phase equilibrium

This avoids gradient degradation through long unrolled chains.
"""

import argparse
from datetime import datetime
import json
import os
from typing import List, Tuple

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import time

import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from jax.tree_util import DictKey
from jax.flatten_util import ravel_pytree

import wandb

from config import Config
from data import load_shakespeare_dataset, prepare_shakespeare_dataset
from model import (
    energy_per_batch,
    get_xi_attn_embed,
    get_xi_hopf,
    get_xi_pos,
    init_params,
    logits_from_v,
)
from utils import (
    ModelParams,
    calculate_perplexity,
    generate_text,
    plot_training_metrics,
    save_metrics,
    save_params,
)


# ---------------------------------------------------------------------------
# EP dynamics
# ---------------------------------------------------------------------------


def run_free_phase(
    params: ModelParams, V0: jax.Array, ctx_bits: jax.Array, cfg: Config
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Run energy gradient descent to approximate equilibrium.

    Returns (V_T, H_attn_T, H_hopf_T, force_T).
    force_T is -(1/tau_v)*dE/dV at the terminal state (for monitoring).
    """
    xi_attn_embed = get_xi_attn_embed(params)
    xi_pos = get_xi_pos(params, ctx_bits.shape[1], V0.shape[1])  # (L, D)
    batch_xi_attn = xi_attn_embed[ctx_bits] + xi_pos[None, :, :]  # (B, L, D)
    xi_hopf = get_xi_hopf(params)  # (M, D)
    step_v = cfg.step_size / cfg.tau_v
    step_h = cfg.step_size / cfg.tau_h

    H_attn0 = jnp.einsum("bld,bd->bl", batch_xi_attn, V0) + params["b"]
    H_hopf0 = V0 @ xi_hopf.T + params["c"]

    def body(_, carry):
        V, H_attn, H_hopf = carry
        F_attn = jax.nn.softmax(cfg.beta * H_attn, axis=-1)
        F_hopf = jnp.maximum(H_hopf, 0.0)

        dE_dV = (
            (V - params["a"])
            - jnp.einsum("bld,bl->bd", batch_xi_attn, F_attn)
            - F_hopf @ xi_hopf
        )
        dE_dF_attn = H_attn - (
            jnp.einsum("bld,bd->bl", batch_xi_attn, V) + params["b"]
        )
        dE_dF_hopf = H_hopf - (V @ xi_hopf.T + params["c"])

        V = V - step_v * dE_dV
        H_attn = H_attn - step_h * dE_dF_attn
        H_hopf = H_hopf - step_h * dE_dF_hopf
        return (V, H_attn, H_hopf)

    V_T, H_attn_T, H_hopf_T = jax.lax.fori_loop(
        0, cfg.n_steps, body, (V0, H_attn0, H_hopf0)
    )

    # Force at terminal state (for monitoring equilibrium quality)
    F_attn_T = jax.nn.softmax(cfg.beta * H_attn_T, axis=-1)
    F_hopf_T = jnp.maximum(H_hopf_T, 0.0)
    dE_dV_T = (
        (V_T - params["a"])
        - jnp.einsum("bld,bl->bd", batch_xi_attn, F_attn_T)
        - F_hopf_T @ xi_hopf
    )
    force_T = -(1.0 / cfg.tau_v) * dE_dV_T

    return V_T, H_attn_T, H_hopf_T, force_T


def run_nudged_phase(
    params: ModelParams,
    V_init: jax.Array,
    H_attn_init: jax.Array,
    H_hopf_init: jax.Array,
    ctx_bits: jax.Array,
    labels: jax.Array,
    beta_nudge: jax.Array,
    v_reg: jax.Array,
    cfg: Config,
    n_nudge_steps: int,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """Run nudged-phase dynamics: energy E + beta * C(V, labels).

    The nudge only affects the V update (C depends on V through the decoder,
    not on H_attn or H_hopf).  Starts from free-phase equilibrium.

    Returns (V_T, H_attn_T, H_hopf_T).
    """
    xi_attn_embed = get_xi_attn_embed(params)
    xi_pos = get_xi_pos(params, ctx_bits.shape[1], V_init.shape[1])  # (L, D)
    batch_xi_attn = xi_attn_embed[ctx_bits] + xi_pos[None, :, :]
    xi_hopf = get_xi_hopf(params)
    step_v = cfg.step_size / cfg.tau_v
    step_h = cfg.step_size / cfg.tau_h

    W_dec = params["W_dec"]
    b_dec = params["b_dec"]

    def body(_, carry):
        V, H_attn, H_hopf = carry
        F_attn = jax.nn.softmax(cfg.beta * H_attn, axis=-1)
        F_hopf = jnp.maximum(H_hopf, 0.0)

        # Energy gradients (same as free phase)
        dE_dV = (
            (V - params["a"])
            - jnp.einsum("bld,bl->bd", batch_xi_attn, F_attn)
            - F_hopf @ xi_hopf
        )
        dE_dF_attn = H_attn - (
            jnp.einsum("bld,bd->bl", batch_xi_attn, V) + params["b"]
        )
        dE_dF_hopf = H_hopf - (V @ xi_hopf.T + params["c"])

        # Nudge: beta * d(mean CE)/dV
        logits = V @ W_dec.T + b_dec  # (B, vocab_size)
        probs = jax.nn.softmax(logits, axis=-1)
        one_hot_y = jax.nn.one_hot(labels, cfg.vocab_size)
        dC_dV = (probs - one_hot_y) @ W_dec / V.shape[0]  # (B, D)
        # Optional stabilizer: 0.5 * v_reg * mean(||V||^2)
        # d/dV = v_reg * V / B
        dC_dV = dC_dV + v_reg * V / V.shape[0]

        V = V - step_v * (dE_dV + beta_nudge * dC_dV)
        H_attn = H_attn - step_h * dE_dF_attn
        H_hopf = H_hopf - step_h * dE_dF_hopf
        return (V, H_attn, H_hopf)

    V_T, H_attn_T, H_hopf_T = jax.lax.fori_loop(
        0, n_nudge_steps, body, (V_init, H_attn_init, H_hopf_init)
    )
    return V_T, H_attn_T, H_hopf_T


def compute_ep_grads(
    params: ModelParams,
    ctx_bits: jax.Array,
    labels: jax.Array,
    V_free: jax.Array,
    H_attn_free: jax.Array,
    H_hopf_free: jax.Array,
    V_nudge: jax.Array,
    H_attn_nudge: jax.Array,
    H_hopf_nudge: jax.Array,
    beta_nudge: jax.Array,
    cfg: Config,
) -> ModelParams:
    """Compute EP parameter gradients.

    Energy params (xi, a, b, c):
        grad = (1/beta) * (dE/dtheta|nudged - dE/dtheta|free)

    Decoder params (W_dec, b_dec):
        grad = d(mean CE)/dtheta evaluated at V_free (standard gradient)

    These two sets are disjoint, so we sum them (zeros where inactive).
    """

    def energy_wrt_params(p, V, H_attn, H_hopf):
        F_attn = jax.nn.softmax(cfg.beta * H_attn, axis=-1)
        F_hopf = jnp.maximum(H_hopf, 0.0)
        return energy_per_batch(p, V, H_attn, H_hopf, F_attn, F_hopf, ctx_bits, cfg)

    dE_free = jax.grad(energy_wrt_params)(
        params, V_free, H_attn_free, H_hopf_free
    )
    dE_nudge = jax.grad(energy_wrt_params)(
        params, V_nudge, H_attn_nudge, H_hopf_nudge
    )

    # EP gradient: (1/beta)(dE|nudge - dE|free)
    # Non-zero only for energy params (xi_attn_embed_raw, xi_hopf_raw, a, b, c)
    ep_grads = jax.tree.map(
        lambda n, f: (n - f) / beta_nudge, dE_nudge, dE_free
    )

    # Standard gradient for decoder at V_free (V_free treated as constant)
    # Non-zero only for W_dec, b_dec
    def decoder_cost(p):
        logits = logits_from_v(p, V_free)
        return jnp.mean(
            optax.softmax_cross_entropy_with_integer_labels(logits, labels)
        )

    dec_grads = jax.grad(decoder_cost)(params)

    return jax.tree.map(jnp.add, ep_grads, dec_grads)


# ---------------------------------------------------------------------------
# Evaluation (uses free-phase dynamics, no BPTT needed)
# ---------------------------------------------------------------------------


def ep_evaluate(
    params: ModelParams, valid_X: jax.Array, valid_y: jax.Array, cfg: Config
) -> Tuple[jax.Array, float]:
    """Evaluate accuracy using free-phase equilibrium."""
    B = valid_y.shape[0]
    V0 = jnp.zeros((B, cfg.D), jnp.float32)
    V_T, _, _, force_T = run_free_phase(params, V0, valid_X, cfg)
    preds = jnp.argmax(logits_from_v(params, V_T), axis=1)
    acc = jnp.mean((preds == valid_y).astype(jnp.float32))
    force_mag = jnp.mean(jnp.sum(force_T * force_T, axis=1))
    return acc, float(force_mag)


def ep_perplexity(
    params: ModelParams, valid_X: jax.Array, valid_y: jax.Array, cfg: Config
) -> float:
    """Calculate perplexity at free-phase equilibrium."""
    B = valid_y.shape[0]
    V0 = jnp.zeros((B, cfg.D), jnp.float32)
    V_T, _, _, _ = run_free_phase(params, V0, valid_X, cfg)
    logits = logits_from_v(params, V_T)
    ce = optax.softmax_cross_entropy_with_integer_labels(logits, valid_y)
    return float(jnp.exp(jnp.mean(ce)))


# ---------------------------------------------------------------------------
# Spectral projection — enforce energy stability condition
# ---------------------------------------------------------------------------
#
# WHY THIS IS NEEDED (not a regularization hack — a physical requirement):
#
# The full energy is a saddle function over (V, H_attn, H_hopf).  After
# eliminating the hidden variables at their equilibrium values, we get a
# *reduced* energy over V alone.  For the Hopfield (ReLU) hidden layer,
# the reduced energy in the all-active region works out to:
#
#   E_hopf_reduced(V) = -0.5 ||xi_hopf V + c||^2
#
# (This follows from substituting h_hopf = xi_hopf V + c, f_hopf = ReLU(h),
#  and the Fenchel conjugate L_hopf(h) = 0.5 ||ReLU(h)||^2 into the energy.)
#
# Combined with the visible quadratic, the total reduced energy behaves as:
#
#   E_red(V) = 0.5 ||V - a||^2 - 0.5 ||xi_hopf V + c||^2
#              - (bounded softmax/attention terms)
#
# For large V, this is dominated by the quadratic:
#
#   E_red ~ 0.5 V^T (I - xi_hopf^T xi_hopf) V + ...
#
# The attention contribution adds a term bounded by:
#
#   -beta * V^T (xi_attn^T J_softmax xi_attn) V
#
# where J_softmax has eigenvalues in [0, 1] (it's diag(p) - p p^T for
# softmax output p).  So the full Hessian of E_red is:
#
#   H = I - xi_hopf^T xi_hopf - beta * xi_attn^T J_softmax xi_attn
#
# For E_red to be bounded below (i.e., for a finite-energy ground state
# to exist), we need H to be positive definite.  A sufficient condition is:
#
#   sigma_max(xi_hopf)^2 + beta * sigma_max(xi_attn)^2 < 1
#
# If this condition is violated, the energy is unbounded below along the
# top singular vector direction — the dynamics will diverge because there
# is literally no equilibrium to converge to.
#
# We enforce this as a hard constraint via projected gradient descent:
# after each optimizer step, if the weights violate the condition, we
# rescale both coupling matrices by a common factor to land on the
# constraint boundary (with a safety margin).
#
# We use power iteration (not SVD) to estimate sigma_max because SVD is
# not available on Metal/GPU backends.


def sigma_max_power_iter(W: jax.Array, n_iters: int = 10) -> jax.Array:
    """Estimate the largest singular value of W via power iteration.

    For a matrix W of shape (M, D), the largest singular value sigma_max
    satisfies:
        sigma_max = max_u ||W u|| / ||u||

    Power iteration finds it by alternating:
        v <- W u / ||W u||      (left singular vector estimate)
        u <- W^T v / ||W^T v||  (right singular vector estimate)

    After convergence, sigma_max ≈ ||W u|| = v^T W u.

    For the 16x16 and 65x16 matrices in this model, 10 iterations gives
    a very accurate estimate (typically < 1e-6 relative error).
    """
    # Deterministic initialization (ones vector, normalized)
    D = W.shape[1]
    u = jnp.ones(D, dtype=W.dtype) / jnp.sqrt(D)

    def body(_, u):
        v = W @ u
        # If W is (close to) all zeros, norms can underflow to 0; avoid 0/0 NaNs.
        v = v / jnp.maximum(jnp.linalg.norm(v), jnp.asarray(1e-12, dtype=W.dtype))
        u = W.T @ v
        u = u / jnp.maximum(jnp.linalg.norm(u), jnp.asarray(1e-12, dtype=W.dtype))
        return u

    u = jax.lax.fori_loop(0, n_iters, body, u)
    return jnp.linalg.norm(W @ u)


def project_weights(
    params: ModelParams, beta: float, margin: float = 0.1
) -> Tuple[ModelParams, jax.Array, jax.Array, jax.Array]:
    """Project coupling weights to satisfy the energy stability condition.

    The condition for the energy to have a finite minimum (ground state) is:

        sigma_max(xi_hopf)^2 + beta * sigma_max(xi_attn)^2 < 1

    where xi_hopf = xi_hopf_raw^2 and xi_attn = xi_attn_embed_raw^2
    (elementwise square for positivity).

    If violated, we rescale both raw parameter matrices by a common factor
    so that the condition holds with the specified safety margin:

        sigma_max(xi_hopf)^2 + beta * sigma_max(xi_attn)^2 = 1 - margin

    DERIVATION OF THE RESCALING:

    Let alpha be the scale factor applied to xi (the squared weights).
    If we scale raw -> raw * sqrt(alpha), then:
        xi_new = (raw * sqrt(alpha))^2 = raw^2 * alpha = xi * alpha

    So sigma_max(xi_new) = alpha * sigma_max(xi).  The new condition:

        (alpha * s_h)^2 + beta * (alpha * s_a)^2
            = alpha^2 * (s_h^2 + beta * s_a^2)
            = 1 - margin

    Solving:  alpha = sqrt((1 - margin) / (s_h^2 + beta * s_a^2))

    Then raw is scaled by sqrt(alpha) so that xi = raw^2 scales by alpha.

    Returns (params, sigma_hopf, sigma_attn, clipped) where clipped is 1.0
    if projection was applied, 0.0 otherwise.
    """
    xi_hopf = get_xi_hopf(params)   # xi_hopf_raw^2, shape (M, D)
    xi_attn = get_xi_attn_embed(params)  # xi_attn_embed_raw^2, shape (vocab, D)

    s_h = sigma_max_power_iter(xi_hopf)
    s_a = sigma_max_power_iter(xi_attn)

    violation = s_h ** 2 + beta * s_a ** 2 - (1.0 - margin)

    # alpha: scale factor on xi (squared weights) to hit the constraint boundary
    # raw_scale: corresponding scale on the raw (pre-square) parameters
    alpha = jnp.sqrt((1.0 - margin) / (s_h ** 2 + beta * s_a ** 2))
    raw_scale = jnp.sqrt(alpha)

    # Only rescale if the constraint is violated (violation > 0)
    do_clip = violation > 0.0
    raw_scale = jnp.where(do_clip, raw_scale, 1.0)

    new_params = dict(params)
    new_params["xi_hopf_raw"] = params["xi_hopf_raw"] * raw_scale
    new_params["xi_attn_embed_raw"] = params["xi_attn_embed_raw"] * raw_scale

    return new_params, s_h, s_a, do_clip.astype(jnp.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Shakespeare model with Equilibrium Propagation"
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--gen_chars", type=int, default=32)
    parser.add_argument("--sample_mode", action="store_true")
    # EP-specific arguments
    parser.add_argument(
        "--beta_nudge",
        type=float,
        default=0.5,
        help="Nudge strength for equilibrium propagation",
    )
    parser.add_argument(
        "--nudge_frac",
        type=float,
        default=0.5,
        help="Fraction of n_steps used for the nudged phase (starts from equilibrium)",
    )
    parser.add_argument(
        "--v_reg",
        type=float,
        default=0.0,
        help="Optional V^2 stabilizer added to the nudged cost: 0.5 * v_reg * mean(||V||^2).",
    )
    parser.add_argument(
        "--ep_gate",
        action="store_true",
        help="Enable EP validity gating: skip parameter updates when equilibrium diagnostics look invalid.",
    )
    parser.add_argument(
        "--ep_on_invalid",
        type=str,
        default="skip",
        choices=["skip", "decoder_only"],
        help="What to do when EP validity checks fail (only active when --ep_gate is set).",
    )
    parser.add_argument(
        "--ep_free_force_thresh",
        type=float,
        default=1e-3,
        help="If free-phase force_mag exceeds this, mark EP invalid and (optionally) skip the update.",
    )
    parser.add_argument(
        "--ep_force_rel_thresh",
        type=float,
        default=None,
        help="If set, gate on the scale-invariant free-phase residual force_rel (recommended).",
    )
    parser.add_argument(
        "--ep_force_rms_thresh",
        type=float,
        default=None,
        help="If set, also accept steps where free-phase force_rms <= thresh (useful when V is tiny).",
    )
    parser.add_argument(
        "--ep_nudge_force_thresh",
        type=float,
        default=1e-3,
        help="If nudged-phase force_nudge_mag exceeds this, mark EP invalid and (optionally) skip the update.",
    )
    parser.add_argument(
        "--ep_disp_thresh",
        type=float,
        default=1.0,
        help="If ||V_nudge - V_free|| exceeds this, mark EP invalid and (optionally) skip the update.",
    )
    parser.add_argument(
        "--ep_disp_rel_thresh",
        type=float,
        default=None,
        help="If set, gate on the scale-invariant displacement disp_rel (recommended).",
    )
    parser.add_argument(
        "--ep_disp_rms_thresh",
        type=float,
        default=None,
        help="If set, also accept steps where V_disp_rms <= thresh (useful when V is tiny).",
    )
    parser.add_argument(
        "--ep_shrink_on_invalid",
        action="store_true",
        help="If EP is invalid, shrink coupling params (xi_*_raw) by ep_shrink_factor to recover.",
    )
    parser.add_argument(
        "--ep_shrink_factor",
        type=float,
        default=0.99,
        help="Multiplicative factor applied to xi_*_raw when EP is invalid and --ep_shrink_on_invalid is set.",
    )
    parser.add_argument(
        "--fast_opt",
        type=str,
        default="adamw",
        choices=["adamw", "sgd"],
        help="Optimizer for energy coupling params (xi_*_raw).",
    )
    parser.add_argument(
        "--fast_lr_mult",
        type=float,
        default=1.0,
        help="Multiplier on lr_peak_value for the fast optimizer schedule.",
    )
    parser.add_argument(
        "--dec_weight_decay",
        type=float,
        default=0.0,
        help="Decoder (W_dec, b_dec) AdamW weight decay. 0.0 recommended to prevent V/W scale drift.",
    )
    parser.add_argument(
        "--dec_lr_mult",
        type=float,
        default=1.0,
        help="Multiplier on lr_peak_value for the decoder optimizer schedule.",
    )
    parser.add_argument(
        "--proj_margin",
        type=float,
        default=0.1,
        help="Safety margin for spectral projection (constraint target = 1 - margin)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run a few steps with detailed per-parameter diagnostics, no W&B",
    )
    parser.add_argument(
        "--debug_steps",
        type=int,
        default=5,
        help="Number of steps to run in debug mode (default: 5)",
    )
    parser.add_argument(
        "--debug_log_path",
        type=str,
        default=None,
        help="If set in debug mode, write per-step diagnostics as JSONL to this path.",
    )
    parser.add_argument(
        "--debug_print_every",
        type=int,
        default=1,
        help="In debug mode, print a one-line summary every N steps (default: 1).",
    )
    parser.add_argument(
        "--debug_full_table",
        action="store_true",
        help="In debug mode, print the per-parameter diagnostics table (very verbose).",
    )
    parser.add_argument(
        "--sanity_check",
        action="store_true",
        help="Run a one-batch EP-vs-BPTT gradient sanity check and exit.",
    )
    parser.add_argument(
        "--sanity_steps",
        type=int,
        default=200,
        help="Number of free-phase steps for sanity check (default: 200).",
    )
    parser.add_argument(
        "--sanity_nudge_steps",
        type=int,
        default=100,
        help="Number of nudged-phase steps for sanity check (default: 100).",
    )
    parser.add_argument(
        "--sanity_batch_size",
        type=int,
        default=8,
        help="Batch size used for sanity check (default: 8).",
    )
    args = parser.parse_args()

    # ---- Load data ----
    filename_prefix = "shakespeare_data"
    dataset_exists = os.path.exists(f"data/{filename_prefix}_train_X.txt")

    if dataset_exists:
        print("Loading Shakespeare dataset from saved files...")
        train_X, train_y, valid_X, valid_y, char_to_idx, idx_to_char = (
            load_shakespeare_dataset(filename_prefix=filename_prefix)
        )
    else:
        print("Preparing Shakespeare dataset...")
        train_X, train_y, valid_X, valid_y, char_to_idx, idx_to_char = (
            prepare_shakespeare_dataset(
                ctx_length=16, train_ratio=0.9, filename_prefix=filename_prefix
            )
        )

    vocab_size = len(char_to_idx)
    print(f"Using vocab_size={vocab_size}")

    if args.sample_mode:
        max_samples = 1000 * 256
        if len(train_X) > max_samples:
            print(f"Sample mode: Using {max_samples} training samples")
            train_X = train_X[:max_samples]
            train_y = train_y[:max_samples]
        if len(valid_X) > max_samples:
            valid_X = valid_X[:max_samples]
            valid_y = valid_y[:max_samples]

    # ---- Config ----
    if args.config:
        print(f"Loading config from {args.config}")
        with open(args.config, "r") as f:
            config_dict = json.load(f)
        shakespeare_config = Config(
            L=config_dict.get("L", 16),
            vocab_size=vocab_size,
            **{k: v for k, v in config_dict.items() if k not in ["L", "vocab_size"]},
        )
    else:
        shakespeare_config = Config(L=16, vocab_size=vocab_size)

    n_nudge_steps = max(1, int(shakespeare_config.n_steps * args.nudge_frac))
    proj_margin = args.proj_margin
    print(
        f"EP config: beta_nudge={args.beta_nudge}, "
        f"free steps={shakespeare_config.n_steps}, nudge steps={n_nudge_steps}, "
        f"proj_margin={proj_margin}"
    )
    v_reg = jnp.asarray(args.v_reg, dtype=jnp.float32)

    # ---- Output dir ----
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"data/shakespeare_ep/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    config_save_dict = {
        k: getattr(shakespeare_config, k)
        for k in dir(shakespeare_config)
        if not k.startswith("_") and not callable(getattr(shakespeare_config, k))
    }
    config_save_dict["sample_mode"] = args.sample_mode
    config_save_dict["temperature"] = args.temperature
    config_save_dict["gen_chars"] = args.gen_chars
    config_save_dict["beta_nudge"] = args.beta_nudge
    config_save_dict["nudge_frac"] = args.nudge_frac
    config_save_dict["n_nudge_steps"] = n_nudge_steps
    config_save_dict["training_method"] = "equilibrium_propagation"
    config_save_dict["proj_margin"] = proj_margin
    config_save_dict["v_reg"] = float(args.v_reg)
    config_save_dict["fast_opt"] = args.fast_opt
    config_save_dict["fast_lr_mult"] = float(args.fast_lr_mult)
    config_save_dict["dec_weight_decay"] = float(args.dec_weight_decay)
    config_save_dict["dec_lr_mult"] = float(args.dec_lr_mult)
    config_save_dict["ep_gate"] = bool(args.ep_gate)
    config_save_dict["ep_on_invalid"] = str(args.ep_on_invalid)
    config_save_dict["ep_free_force_thresh"] = float(args.ep_free_force_thresh)
    config_save_dict["ep_force_rel_thresh"] = (
        None if args.ep_force_rel_thresh is None else float(args.ep_force_rel_thresh)
    )
    config_save_dict["ep_force_rms_thresh"] = (
        None if args.ep_force_rms_thresh is None else float(args.ep_force_rms_thresh)
    )
    config_save_dict["ep_nudge_force_thresh"] = float(args.ep_nudge_force_thresh)
    config_save_dict["ep_disp_thresh"] = float(args.ep_disp_thresh)
    config_save_dict["ep_disp_rel_thresh"] = (
        None if args.ep_disp_rel_thresh is None else float(args.ep_disp_rel_thresh)
    )
    config_save_dict["ep_disp_rms_thresh"] = (
        None if args.ep_disp_rms_thresh is None else float(args.ep_disp_rms_thresh)
    )
    config_save_dict["ep_shrink_on_invalid"] = bool(args.ep_shrink_on_invalid)
    config_save_dict["ep_shrink_factor"] = float(args.ep_shrink_factor)
    if args.config:
        config_save_dict["config_file"] = args.config

    with open(f"{output_dir}/run_config.json", "w") as f:
        json.dump(config_save_dict, f, indent=2)

    # ---- W&B ----
    if not args.debug:
        try:
            wb_config = vars(shakespeare_config)
        except TypeError:
            wb_config = {
                k: getattr(shakespeare_config, k)
                for k in dir(shakespeare_config)
                if not k.startswith("_")
                and isinstance(getattr(shakespeare_config, k), (int, float, str, bool))
            }
        wb_config["beta_nudge"] = args.beta_nudge
        wb_config["nudge_frac"] = args.nudge_frac
        wb_config["training_method"] = "equilibrium_propagation"
        wb_config["proj_margin"] = proj_margin
        wb_config["v_reg"] = float(args.v_reg)
        wb_config["fast_opt"] = args.fast_opt
        wb_config["fast_lr_mult"] = float(args.fast_lr_mult)
        wb_config["dec_weight_decay"] = float(args.dec_weight_decay)
        wb_config["dec_lr_mult"] = float(args.dec_lr_mult)
        wb_config["ep_gate"] = bool(args.ep_gate)
        wb_config["ep_on_invalid"] = str(args.ep_on_invalid)
        wb_config["ep_free_force_thresh"] = float(args.ep_free_force_thresh)
        wb_config["ep_force_rel_thresh"] = (
            None if args.ep_force_rel_thresh is None else float(args.ep_force_rel_thresh)
        )
        wb_config["ep_force_rms_thresh"] = (
            None if args.ep_force_rms_thresh is None else float(args.ep_force_rms_thresh)
        )
        wb_config["ep_nudge_force_thresh"] = float(args.ep_nudge_force_thresh)
        wb_config["ep_disp_thresh"] = float(args.ep_disp_thresh)
        wb_config["ep_disp_rel_thresh"] = (
            None if args.ep_disp_rel_thresh is None else float(args.ep_disp_rel_thresh)
        )
        wb_config["ep_disp_rms_thresh"] = (
            None if args.ep_disp_rms_thresh is None else float(args.ep_disp_rms_thresh)
        )
        wb_config["ep_shrink_on_invalid"] = bool(args.ep_shrink_on_invalid)
        wb_config["ep_shrink_factor"] = float(args.ep_shrink_factor)

        run = wandb.init(
            entity="qpaig",
            project="analog-et",
            name=f"EP_ctx{shakespeare_config.L}_bs{shakespeare_config.batch_size}",
            config=wb_config,
        )

    # ---- Init ----
    key = jr.PRNGKey(shakespeare_config.seed)
    params = init_params(key, shakespeare_config)

    # ---- Enforce stability constraint at initialization ----
    params, s_h_init, s_a_init, clipped_init = project_weights(
        params, shakespeare_config.beta, margin=proj_margin
    )
    if float(clipped_init) > 0.0:
        print(
            "Initial projection applied: "
            f"sigma_hopf={float(s_h_init):.4f}, "
            f"sigma_attn={float(s_a_init):.4f}, "
            f"margin={proj_margin}"
        )

    num_train = train_X.shape[0]
    num_batches = (num_train + shakespeare_config.batch_size - 1) // shakespeare_config.batch_size
    total_steps = shakespeare_config.max_steps

    def lr_sched(peak: float, warmup_steps: int = 100) -> optax.Schedule:
        return optax.warmup_cosine_decay_schedule(
            init_value=shakespeare_config.lr_init_value,
            peak_value=peak,
            warmup_steps=warmup_steps,
            decay_steps=max(1, total_steps - warmup_steps),
            end_value=peak * shakespeare_config.lr_end_factor,
        )

    lr_schedule_fast = lr_sched(shakespeare_config.lr_peak_value * args.fast_lr_mult)
    lr_schedule_slow = lr_sched(shakespeare_config.lr_peak_value)
    lr_schedule_dec = lr_sched(shakespeare_config.lr_peak_value * args.dec_lr_mult)

    if args.fast_opt == "sgd":
        tx_fast = optax.chain(
            optax.clip_by_global_norm(shakespeare_config.max_norm),
            optax.sgd(learning_rate=lr_schedule_fast, momentum=0.0),
        )
    else:
        tx_fast = optax.chain(
            optax.clip_by_global_norm(shakespeare_config.max_norm),
            optax.adamw(
                learning_rate=lr_schedule_fast,
                weight_decay=shakespeare_config.fast_weight_decay,
            ),
        )
    tx_slow = optax.chain(
        optax.clip_by_global_norm(shakespeare_config.max_norm),
        optax.adamw(
            learning_rate=lr_schedule_slow,
            weight_decay=shakespeare_config.slow_weight_decay,
        ),
    )
    tx_dec = optax.chain(
        optax.clip_by_global_norm(shakespeare_config.max_norm),
        optax.adamw(
            learning_rate=lr_schedule_dec,
            weight_decay=args.dec_weight_decay,
        ),
    )

    # Parameter partition for EP: fast couplings, decoder head, everything else.
    def label_tree_ep(params: ModelParams):
        def label_from_path(path, leaf) -> str:
            last = path[-1]
            if isinstance(last, DictKey):
                name = last.key
            else:
                name = str(last)
            if name in ("xi_attn_embed_raw", "xi_hopf_raw"):
                return "fast"
            if name in ("W_dec", "b_dec"):
                return "dec"
            return "slow"

        return jax.tree_util.tree_map_with_path(label_from_path, params)

    optimizer = optax.multi_transform(
        {"fast": tx_fast, "slow": tx_slow, "dec": tx_dec}, label_tree_ep(params)
    )
    opt_state = optimizer.init(params)

    # =====================================================================
    # SANITY CHECK: EP GRADIENT VS BPTT GRADIENT (ONE BATCH)
    # =====================================================================
    if args.sanity_check:
        # Smaller config for BPTT unroll
        sanity_cfg = Config(
            D=shakespeare_config.D,
            L=shakespeare_config.L,
            M=shakespeare_config.M,
            beta=shakespeare_config.beta,
            xi_attn_embed_raw_scale=shakespeare_config.xi_attn_embed_raw_scale,
            xi_hopf_raw_scale=shakespeare_config.xi_hopf_raw_scale,
            step_size=shakespeare_config.step_size,
            T_final=float(args.sanity_steps) * shakespeare_config.step_size,
            batch_size=args.sanity_batch_size,
            max_steps=1,
            seed=shakespeare_config.seed,
            vocab_size=shakespeare_config.vocab_size,
            tau_v=shakespeare_config.tau_v,
            tau_h=shakespeare_config.tau_h,
            lr_init_value=shakespeare_config.lr_init_value,
            lr_peak_value=shakespeare_config.lr_peak_value,
            lr_end_factor=shakespeare_config.lr_end_factor,
            max_norm=shakespeare_config.max_norm,
            slow_weight_decay=shakespeare_config.slow_weight_decay,
            fast_weight_decay=shakespeare_config.fast_weight_decay,
            force_penalty_start=shakespeare_config.force_penalty_start,
            force_penalty_duration=shakespeare_config.force_penalty_duration,
            force_penalty_scale=shakespeare_config.force_penalty_scale,
        )

        beta_val = jnp.asarray(args.beta_nudge, dtype=jnp.float32)
        bx = train_X[: args.sanity_batch_size]
        by = train_y[: args.sanity_batch_size]

        def bptt_loss(p):
            V0 = jnp.zeros((bx.shape[0], sanity_cfg.D), dtype=jnp.float32)
            V_T, _, _, _ = run_free_phase(p, V0, bx, sanity_cfg)
            logits = logits_from_v(p, V_T)
            return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, by))

        grads_bptt = jax.grad(bptt_loss)(params)

        # EP grads using the same (short) dynamics lengths.
        def ep_grads_one_batch(p):
            B = bx.shape[0]
            V0 = jnp.zeros((B, sanity_cfg.D), dtype=jnp.float32)
            V_free, H_attn_free, H_hopf_free, _ = run_free_phase(p, V0, bx, sanity_cfg)
            V_nudge, H_attn_nudge, H_hopf_nudge = run_nudged_phase(
                p,
                V_free,
                H_attn_free,
                H_hopf_free,
                bx,
                by,
                beta_val,
                v_reg,
                sanity_cfg,
                int(args.sanity_nudge_steps),
            )

            def energy_wrt_params(pp, V, H_attn, H_hopf):
                F_attn = jax.nn.softmax(sanity_cfg.beta * H_attn, axis=-1)
                F_hopf = jnp.maximum(H_hopf, 0.0)
                return energy_per_batch(pp, V, H_attn, H_hopf, F_attn, F_hopf, bx, sanity_cfg)

            dE_free = jax.grad(energy_wrt_params)(p, V_free, H_attn_free, H_hopf_free)
            dE_nudge = jax.grad(energy_wrt_params)(p, V_nudge, H_attn_nudge, H_hopf_nudge)
            ep_g = jax.tree.map(lambda n, f: (n - f) / beta_val, dE_nudge, dE_free)

            def dec_cost(pp):
                logits = logits_from_v(pp, V_free)
                return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, by))

            dec_g = jax.grad(dec_cost)(p)
            return jax.tree.map(jnp.add, ep_g, dec_g)

        grads_ep = ep_grads_one_batch(params)

        # Compare cosine similarity of the flattened gradients.
        v_ep, _ = ravel_pytree(grads_ep)
        v_bp, _ = ravel_pytree(grads_bptt)
        denom = (jnp.linalg.norm(v_ep) * jnp.linalg.norm(v_bp)) + 1e-12
        cos = jnp.vdot(v_ep, v_bp) / denom

        print("\n" + "=" * 72)
        print("SANITY CHECK: EP vs BPTT gradient (one batch)")
        print("=" * 72)
        print(f"sanity_steps       = {args.sanity_steps}")
        print(f"sanity_nudge_steps = {args.sanity_nudge_steps}")
        print(f"beta_nudge         = {float(args.beta_nudge)}")
        print(f"cosine_similarity  = {float(cos):.6f}")
        print(f"||g_ep||           = {float(jnp.linalg.norm(v_ep)):.6e}")
        print(f"||g_bptt||         = {float(jnp.linalg.norm(v_bp)):.6e}")
        print("=" * 72 + "\n")

        import sys
        sys.exit(0)

    # ---- Train step (returns diagnostics dict for debug) ----

    def _ep_forward(params, batch_X, batch_Y, beta):
        """Run both EP phases and compute gradients. Returns everything needed
        for both the optimizer step and debug printing."""
        B = batch_X.shape[0]
        V0 = jnp.zeros((B, shakespeare_config.D))

        xi_attn_embed = get_xi_attn_embed(params)
        batch_xi_attn = xi_attn_embed[batch_X]  # (B, L, D)
        xi_hopf = get_xi_hopf(params)  # (M, D)

        # Free phase
        V_free, H_attn_free, H_hopf_free, force_free = run_free_phase(
            params, V0, batch_X, shakespeare_config
        )
        V_free = jax.lax.stop_gradient(V_free)
        H_attn_free = jax.lax.stop_gradient(H_attn_free)
        H_hopf_free = jax.lax.stop_gradient(H_hopf_free)

        # Nudged phase
        V_nudge, H_attn_nudge, H_hopf_nudge = run_nudged_phase(
            params, V_free, H_attn_free, H_hopf_free,
            batch_X, batch_Y, beta, v_reg, shakespeare_config, n_nudge_steps,
        )
        V_nudge = jax.lax.stop_gradient(V_nudge)
        H_attn_nudge = jax.lax.stop_gradient(H_attn_nudge)
        H_hopf_nudge = jax.lax.stop_gradient(H_hopf_nudge)

        # ------------------------------------------------------------------
        # Equilibrium residual diagnostics (free and nudged phases)
        # ------------------------------------------------------------------
        # Hidden residuals: these are exactly the update directions used for H
        # in our dynamics (dual/mirror-coordinates), so they should be ~0 at a
        # fixed point.
        free_attn_resid = H_attn_free - (
            jnp.einsum("bld,bd->bl", batch_xi_attn, V_free) + params["b"]
        )
        free_hopf_resid = H_hopf_free - (V_free @ xi_hopf.T + params["c"])
        free_attn_resid_mag = jnp.mean(
            jnp.sum(free_attn_resid * free_attn_resid, axis=1)
        )
        free_hopf_resid_mag = jnp.mean(
            jnp.sum(free_hopf_resid * free_hopf_resid, axis=1)
        )

        nudge_attn_resid = H_attn_nudge - (
            jnp.einsum("bld,bd->bl", batch_xi_attn, V_nudge) + params["b"]
        )
        nudge_hopf_resid = H_hopf_nudge - (V_nudge @ xi_hopf.T + params["c"])
        nudge_attn_resid_mag = jnp.mean(
            jnp.sum(nudge_attn_resid * nudge_attn_resid, axis=1)
        )
        nudge_hopf_resid_mag = jnp.mean(
            jnp.sum(nudge_hopf_resid * nudge_hopf_resid, axis=1)
        )

        # Visible residual for the nudged fixed point: this should be ~0 when
        # the nudged phase has converged.
        F_attn_n = jax.nn.softmax(shakespeare_config.beta * H_attn_nudge, axis=-1)
        F_hopf_n = jnp.maximum(H_hopf_nudge, 0.0)
        dE_dV_n = (
            (V_nudge - params["a"])
            - jnp.einsum("bld,bl->bd", batch_xi_attn, F_attn_n)
            - F_hopf_n @ xi_hopf
        )
        logits_n = V_nudge @ params["W_dec"].T + params["b_dec"]
        probs_n = jax.nn.softmax(logits_n, axis=-1)
        one_hot_y = jax.nn.one_hot(batch_Y, shakespeare_config.vocab_size)
        dC_dV_n = (probs_n - one_hot_y) @ params["W_dec"] / B
        dC_dV_n = dC_dV_n + v_reg * V_nudge / B
        force_nudge = -(1.0 / shakespeare_config.tau_v) * (dE_dV_n + beta * dC_dV_n)
        force_nudge_mag = jnp.mean(jnp.sum(force_nudge * force_nudge, axis=1))

        # --- EP gradient breakdown (for debug) ---
        def energy_wrt_params(p, V, H_attn, H_hopf):
            F_attn = jax.nn.softmax(shakespeare_config.beta * H_attn, axis=-1)
            F_hopf = jnp.maximum(H_hopf, 0.0)
            return energy_per_batch(
                p, V, H_attn, H_hopf, F_attn, F_hopf, batch_X, shakespeare_config
            )

        dE_free = jax.grad(energy_wrt_params)(
            params, V_free, H_attn_free, H_hopf_free
        )
        dE_nudge = jax.grad(energy_wrt_params)(
            params, V_nudge, H_attn_nudge, H_hopf_nudge
        )
        ep_grads = jax.tree.map(
            lambda n, f: (n - f) / beta, dE_nudge, dE_free
        )

        def decoder_cost(p):
            logits = logits_from_v(p, V_free)
            return jnp.mean(
                optax.softmax_cross_entropy_with_integer_labels(logits, batch_Y)
            )

        dec_grads = jax.grad(decoder_cost)(params)
        grads = jax.tree.map(jnp.add, ep_grads, dec_grads)

        # CE loss at free equilibrium
        logits = logits_from_v(params, V_free)
        ce_loss = jnp.mean(
            optax.softmax_cross_entropy_with_integer_labels(logits, batch_Y)
        )
        force_mag = jnp.mean(jnp.sum(force_free * force_free, axis=1))

        # Per-parameter diagnostics
        diag = {
            "ce_loss": ce_loss,
            "force_mag": force_mag,
            "force_nudge_mag": force_nudge_mag,
            "free_attn_resid_mag": free_attn_resid_mag,
            "free_hopf_resid_mag": free_hopf_resid_mag,
            "nudge_attn_resid_mag": nudge_attn_resid_mag,
            "nudge_hopf_resid_mag": nudge_hopf_resid_mag,
            # Note: these are Frobenius norms over the whole batch (scale with sqrt(B)).
            "V_free_norm": jnp.linalg.norm(V_free),
            "V_nudge_norm": jnp.linalg.norm(V_nudge),
            "V_displacement": jnp.linalg.norm(V_nudge - V_free),
            "H_attn_free_norm": jnp.linalg.norm(H_attn_free),
            "H_hopf_free_norm": jnp.linalg.norm(H_hopf_free),
        }

        # Scale-invariant versions (per-sample RMS); these are much easier to
        # interpret once V scales up during training.
        V_free_rms = jnp.sqrt(jnp.mean(jnp.sum(V_free * V_free, axis=1)))
        V_disp_rms = jnp.sqrt(jnp.mean(jnp.sum((V_nudge - V_free) ** 2, axis=1)))
        force_rms = jnp.sqrt(force_mag)
        diag["V_free_rms"] = V_free_rms
        diag["V_disp_rms"] = V_disp_rms
        diag["force_rms"] = force_rms
        diag["force_rel"] = force_rms / jnp.maximum(V_free_rms, 1e-8)
        diag["disp_rel"] = V_disp_rms / jnp.maximum(V_free_rms, 1e-8)
        for k in params:
            diag[f"param|{k}"] = jnp.linalg.norm(params[k])
            diag[f"dE_free|{k}"] = jnp.linalg.norm(dE_free[k])
            diag[f"dE_nudge|{k}"] = jnp.linalg.norm(dE_nudge[k])
            diag[f"dE_diff|{k}"] = jnp.linalg.norm(dE_nudge[k] - dE_free[k])
            diag[f"ep_grad|{k}"] = jnp.linalg.norm(ep_grads[k])
            diag[f"dec_grad|{k}"] = jnp.linalg.norm(dec_grads[k])
            diag[f"grad|{k}"] = jnp.linalg.norm(grads[k])

        return grads, diag

    @jax.jit
    def train_step(params, opt_state, batch_X, batch_Y, beta):
        grads, diag = _ep_forward(params, batch_X, batch_Y, beta)

        grad_norm = optax.global_norm(grads)

        # ------------------------------------------------------------------
        # EP validity gating
        # ------------------------------------------------------------------
        free_ok = diag["force_mag"] <= args.ep_free_force_thresh
        if args.ep_force_rel_thresh is not None:
            free_ok = free_ok | (
                diag["force_rel"]
                <= jnp.asarray(args.ep_force_rel_thresh, dtype=jnp.float32)
            )
        if args.ep_force_rms_thresh is not None:
            free_ok = free_ok | (
                diag["force_rms"]
                <= jnp.asarray(args.ep_force_rms_thresh, dtype=jnp.float32)
            )
        nudge_ok = diag["force_nudge_mag"] <= args.ep_nudge_force_thresh
        disp_ok = diag["V_displacement"] <= args.ep_disp_thresh
        if args.ep_disp_rel_thresh is not None:
            disp_ok = disp_ok | (
                diag["disp_rel"]
                <= jnp.asarray(args.ep_disp_rel_thresh, dtype=jnp.float32)
            )
        if args.ep_disp_rms_thresh is not None:
            disp_ok = disp_ok | (
                diag["V_disp_rms"]
                <= jnp.asarray(args.ep_disp_rms_thresh, dtype=jnp.float32)
            )
        finite_ok = jnp.isfinite(diag["ce_loss"]) & jnp.isfinite(grad_norm)
        ep_valid = free_ok & nudge_ok & disp_ok & finite_ok

        ep_gate_enabled = bool(args.ep_gate)
        ep_on_invalid_decoder = args.ep_on_invalid == "decoder_only"

        # Decoder-only gradient mask (used when EP is invalid but we still want
        # supervised learning to keep moving).
        decoder_keys = ("W_dec", "b_dec")
        grads_decoder = {
            k: grads[k] if k in decoder_keys else jnp.zeros_like(grads[k]) for k in grads
        }

        shrink_factor = jnp.asarray(args.ep_shrink_factor, dtype=jnp.float32)

        def _maybe_shrink_fast(p, do_shrink):
            """When EP is invalid, shrink fast coupling params to recover contractivity."""
            def _shrink(pp):
                newp = dict(pp)
                newp["xi_hopf_raw"] = pp["xi_hopf_raw"] * shrink_factor
                newp["xi_attn_embed_raw"] = pp["xi_attn_embed_raw"] * shrink_factor
                return newp

            return jax.lax.cond(do_shrink, _shrink, lambda pp: pp, p)

        def _project_backstop(p):
            p2, s_h, s_a, clipped = project_weights(
                p, shakespeare_config.beta, margin=proj_margin
            )
            constraint_pre = s_h ** 2 + shakespeare_config.beta * s_a ** 2
            constraint_post = jnp.where(clipped > 0.0, 1.0 - proj_margin, constraint_pre)
            return p2, s_h, s_a, constraint_pre, constraint_post, clipped

        def _apply_updates(grads_to_use):
            updates, new_opt_state = optimizer.update(grads_to_use, opt_state, params)
            update_norm = optax.global_norm(updates)
            param_norm = optax.global_norm(params)
            relative_step_size = update_norm / jnp.maximum(param_norm, 1e-8)
            new_params = optax.apply_updates(params, updates)

            do_shrink = args.ep_shrink_on_invalid & (~ep_valid)
            new_params = _maybe_shrink_fast(new_params, do_shrink)

            new_params, s_h, s_a, constraint_pre, constraint_post, clipped = _project_backstop(
                new_params
            )
            return (
                new_params,
                new_opt_state,
                update_norm,
                relative_step_size,
                s_h,
                s_a,
                constraint_pre,
                constraint_post,
                clipped,
                do_shrink.astype(jnp.float32),
            )

        def _skip_updates():
            do_shrink = args.ep_shrink_on_invalid & (~ep_valid)
            new_params = _maybe_shrink_fast(params, do_shrink)
            new_params, s_h, s_a, constraint_pre, constraint_post, clipped = _project_backstop(
                new_params
            )
            return (
                new_params,
                opt_state,
                jnp.asarray(0.0, dtype=jnp.float32),
                jnp.asarray(0.0, dtype=jnp.float32),
                s_h,
                s_a,
                constraint_pre,
                constraint_post,
                clipped,
                do_shrink.astype(jnp.float32),
            )

        def _full_update(_):
            return _apply_updates(grads)

        def _decoder_update(_):
            return _apply_updates(grads_decoder)

        def _invalid_branch(_):
            # 1) decoder_only: keep learning the supervised head while we recover
            # 2) skip: don't update (but still optionally shrink fast couplings)
            # If we have NaNs/Infs, never apply an optimizer update (even decoder-only),
            # otherwise we can corrupt params/opt_state. Still allow shrink/backstop projection.
            def _finite_case(__):
                if ep_on_invalid_decoder:
                    return _decoder_update(None)
                return _skip_updates()

            return jax.lax.cond(finite_ok, _finite_case, lambda __: _skip_updates(), operand=None)

        if ep_gate_enabled:
            (
                params,
                opt_state,
                update_norm,
                relative_step_size,
                s_h,
                s_a,
                constraint_pre,
                constraint_post,
                clipped,
                shrink_applied,
            ) = jax.lax.cond(ep_valid, _full_update, _invalid_branch, operand=None)
        else:
            (
                params,
                opt_state,
                update_norm,
                relative_step_size,
                s_h,
                s_a,
                constraint_pre,
                constraint_post,
                clipped,
                shrink_applied,
            ) = _full_update(None)

        # Encode update mode: 0=full, 1=decoder_only, 2=skip
        mode = jnp.asarray(0.0, dtype=jnp.float32)
        if ep_gate_enabled:
            mode = jnp.where((~ep_valid) & ep_on_invalid_decoder, 1.0, mode)
            mode = jnp.where((~ep_valid) & (~ep_on_invalid_decoder), 2.0, mode)

        diag["ep/valid"] = ep_valid.astype(jnp.float32)
        diag["ep/skipped"] = (ep_gate_enabled & (~ep_valid)).astype(jnp.float32)
        diag["ep/free_ok"] = free_ok.astype(jnp.float32)
        diag["ep/nudge_ok"] = nudge_ok.astype(jnp.float32)
        diag["ep/disp_ok"] = disp_ok.astype(jnp.float32)
        diag["ep/mode"] = mode
        diag["ep/shrink_applied"] = shrink_applied

        diag["grad_norm"] = grad_norm
        diag["update_norm"] = update_norm
        diag["relative_step_size"] = relative_step_size
        diag["proj/sigma_hopf"] = s_h
        diag["proj/sigma_attn"] = s_a
        diag["proj/constraint_pre"] = constraint_pre
        diag["proj/constraint_post"] = constraint_post
        diag["proj/gap_pre"] = (1.0 - proj_margin) - constraint_pre
        diag["proj/gap_post"] = (1.0 - proj_margin) - constraint_post
        diag["proj/applied"] = clipped

        return params, opt_state, diag

    # =====================================================================
    # DEBUG MODE
    # =====================================================================
    if args.debug:
        print("\n" + "=" * 72)
        print("DEBUG MODE — running", args.debug_steps, "steps with diagnostics")
        print("=" * 72)

        beta_val = jnp.asarray(args.beta_nudge, dtype=jnp.float32)

        debug_f = None
        if args.debug_log_path:
            debug_dir = os.path.dirname(args.debug_log_path)
            if debug_dir:
                os.makedirs(debug_dir, exist_ok=True)
            debug_f = open(args.debug_log_path, "w")
            debug_f.write(
                json.dumps(
                    {
                        "meta": {
                            "beta_nudge": float(args.beta_nudge),
                            "nudge_frac": float(args.nudge_frac),
                            "n_nudge_steps": int(n_nudge_steps),
                            "proj_margin": float(proj_margin),
                            "config": config_save_dict,
                        }
                    }
                )
                + "\n"
            )

        for step in range(args.debug_steps):
            start = step * shakespeare_config.batch_size
            if start >= num_train:
                print(
                    f"\nDebug run reached end of training data at step {step} "
                    f"(start={start}, num_train={num_train}). Stopping."
                )
                break
            stop = min(start + shakespeare_config.batch_size, num_train)
            if stop <= start:
                print(
                    f"\nDebug run has empty batch at step {step} "
                    f"(start={start}, stop={stop}). Stopping."
                )
                break
            bx = train_X[start:stop]
            by = train_y[start:stop]

            params, opt_state, diag = train_step(params, opt_state, bx, by, beta_val)

            # Force all values to host for printing
            diag = {k: float(v) for k, v in diag.items()}
            if debug_f is not None:
                debug_f.write(json.dumps({"step": step, **diag}) + "\n")

            if step % max(1, args.debug_print_every) == 0:
                print(f"\n--- step {step} ---")
                print(f"  ce_loss          = {diag['ce_loss']:.6f}")
                print(f"  force_free_mag   = {diag['force_mag']:.6e}")
                print(f"  force_nudge_mag  = {diag['force_nudge_mag']:.6e}")
                print(
                    f"  ep_valid         = {diag.get('ep/valid', 1.0):.1f} "
                    f"(skipped={diag.get('ep/skipped', 0.0):.1f}, "
                    f"free_ok={diag.get('ep/free_ok', 1.0):.1f}, "
                    f"nudge_ok={diag.get('ep/nudge_ok', 1.0):.1f}, "
                    f"disp_ok={diag.get('ep/disp_ok', 1.0):.1f})"
                )
                print(
                    f"  ep_mode          = {diag.get('ep/mode', 0.0):.1f} "
                    f"(shrink={diag.get('ep/shrink_applied', 0.0):.1f})"
                )
                print(f"  grad_norm        = {diag['grad_norm']:.6e}")
                print(f"  update_norm      = {diag['update_norm']:.6e}")
                print(f"  rel_step_size    = {diag['relative_step_size']:.6e}")
                print(f"  proj_applied     = {diag['proj/applied']:.1f}")
                print(f"  constraint       = {diag['proj/constraint_pre']:.6f} -> {diag['proj/constraint_post']:.6f}")
                print(f"  gap              = {diag['proj/gap_pre']:.6f} -> {diag['proj/gap_post']:.6f}")
                print(f"  V_free_norm      = {diag['V_free_norm']:.6f}")
                print(f"  V_nudge_norm     = {diag['V_nudge_norm']:.6f}")
                print(f"  V_displacement   = {diag['V_displacement']:.6e}")
                print(f"  free_attn_resid  = {diag['free_attn_resid_mag']:.6e}")
                print(f"  free_hopf_resid  = {diag['free_hopf_resid_mag']:.6e}")
                print(f"  nudge_attn_resid = {diag['nudge_attn_resid_mag']:.6e}")
                print(f"  nudge_hopf_resid = {diag['nudge_hopf_resid_mag']:.6e}")

            if args.debug_full_table:
                # Per-parameter table
                param_keys = sorted({k.split("|")[1] for k in diag if "|" in k})
                hdr = (
                    f"  {'param':<22s} {'|p|':>10s} {'|dE_f|':>10s} "
                    f"{'|dE_n|':>10s} {'|diff|':>10s} {'|ep_g|':>10s} "
                    f"{'|dec_g|':>10s} {'|grad|':>10s}"
                )
                print(hdr)
                print("  " + "-" * (len(hdr) - 2))
                for k in param_keys:
                    print(
                        f"  {k:<22s}"
                        f" {diag[f'param|{k}']:10.4e}"
                        f" {diag[f'dE_free|{k}']:10.4e}"
                        f" {diag[f'dE_nudge|{k}']:10.4e}"
                        f" {diag[f'dE_diff|{k}']:10.4e}"
                        f" {diag[f'ep_grad|{k}']:10.4e}"
                        f" {diag[f'dec_grad|{k}']:10.4e}"
                        f" {diag[f'grad|{k}']:10.4e}"
                    )

        print("\nDebug run complete.")
        if debug_f is not None:
            debug_f.close()
        import sys
        sys.exit(0)

    # =====================================================================
    # NORMAL TRAINING LOOP
    # =====================================================================
    losses_all: List[float] = []
    losses_steps: List[int] = []
    accs_all: List[float] = []
    accs_steps: List[int] = []
    perplexities_all: List[float] = []
    perplexities_steps: List[int] = []
    generated_texts_all: List[str] = []
    generated_texts_epochs: List[int] = []

    global_step = 0
    seed_context = valid_X[0]
    epoch = 0
    done = False

    beta_val = jnp.asarray(args.beta_nudge, dtype=jnp.float32)

    while not done:
        t_start = time.time()
        key, key_perm = jr.split(key)
        index_perm = jr.permutation(key_perm, num_train)
        train_X_epoch = train_X[index_perm]
        train_y_epoch = train_y[index_perm]
        losses_epoch = []

        for batch in range(num_batches):
            if global_step >= shakespeare_config.max_steps:
                done = True
                break

            start = batch * shakespeare_config.batch_size
            stop = min((batch + 1) * shakespeare_config.batch_size, num_train)
            batch_train_X = train_X_epoch[start:stop]
            batch_train_y = train_y_epoch[start:stop]

            params, opt_state, diag = train_step(
                params, opt_state, batch_train_X, batch_train_y, beta_val
            )

            loss = float(diag["ce_loss"])
            losses_epoch.append(loss)
            losses_all.append(loss)
            losses_steps.append(global_step)

            log_dict = {
                "train/loss": loss,
                "train/grad_norm": float(diag["grad_norm"]),
                "train/update_norm": float(diag["update_norm"]),
                "train/relative_step_size": float(diag["relative_step_size"]),
                "train/force_mag": float(diag["force_mag"]),
                "train/force_nudge_mag": float(diag["force_nudge_mag"]),
                "train/force_rms": float(diag["force_rms"]),
                "train/force_rel": float(diag["force_rel"]),
                "train/free_attn_resid_mag": float(diag["free_attn_resid_mag"]),
                "train/free_hopf_resid_mag": float(diag["free_hopf_resid_mag"]),
                "train/nudge_attn_resid_mag": float(diag["nudge_attn_resid_mag"]),
                "train/nudge_hopf_resid_mag": float(diag["nudge_hopf_resid_mag"]),
                "train/V_free_norm": float(diag["V_free_norm"]),
                "train/V_nudge_norm": float(diag["V_nudge_norm"]),
                "train/V_displacement": float(diag["V_displacement"]),
                "train/V_free_rms": float(diag["V_free_rms"]),
                "train/V_disp_rms": float(diag["V_disp_rms"]),
                "train/disp_rel": float(diag["disp_rel"]),
                "train/H_attn_free_norm": float(diag["H_attn_free_norm"]),
                "train/H_hopf_free_norm": float(diag["H_hopf_free_norm"]),
                "ep/valid": float(diag.get("ep/valid", 1.0)),
                "ep/skipped": float(diag.get("ep/skipped", 0.0)),
                "ep/free_ok": float(diag.get("ep/free_ok", 1.0)),
                "ep/nudge_ok": float(diag.get("ep/nudge_ok", 1.0)),
                "ep/disp_ok": float(diag.get("ep/disp_ok", 1.0)),
                "ep/mode": float(diag.get("ep/mode", 0.0)),
                "ep/shrink_applied": float(diag.get("ep/shrink_applied", 0.0)),
                "proj/sigma_hopf": float(diag["proj/sigma_hopf"]),
                "proj/sigma_attn": float(diag["proj/sigma_attn"]),
                "proj/constraint_pre": float(diag["proj/constraint_pre"]),
                "proj/constraint_post": float(diag["proj/constraint_post"]),
                "proj/applied": float(diag["proj/applied"]),
                "train/beta_nudge": float(beta_val),
                "train/lr_fast": float(lr_schedule_fast(global_step)),
                "train/lr_slow": float(lr_schedule_slow(global_step)),
                "train/lr_dec": float(lr_schedule_dec(global_step)),
                "epoch": epoch,
            }
            for k in params.keys():
                log_dict[f"debug/param_norm/{k}"] = float(diag[f"param|{k}"])
                log_dict[f"debug/dE_free_norm/{k}"] = float(diag[f"dE_free|{k}"])
                log_dict[f"debug/dE_nudge_norm/{k}"] = float(diag[f"dE_nudge|{k}"])
                log_dict[f"debug/dE_diff_norm/{k}"] = float(diag[f"dE_diff|{k}"])
                log_dict[f"debug/ep_grad_norm/{k}"] = float(diag[f"ep_grad|{k}"])
                log_dict[f"debug/dec_grad_norm/{k}"] = float(diag[f"dec_grad|{k}"])
                log_dict[f"debug/grad_norm/{k}"] = float(diag[f"grad|{k}"])

            wandb.log(
                log_dict,
                step=global_step,
            )

            global_step += 1

        t_end = time.time()
        epoch_time = t_end - t_start

        if not losses_epoch:
            break

        mean_train_loss = float(jnp.mean(jnp.array(losses_epoch)))

        wandb.log(
            {
                "epoch/train_loss_mean": mean_train_loss,
                "epoch/time_sec": float(epoch_time),
                "epoch": epoch,
            },
            step=global_step - 1,
        )

        acc, eval_force = ep_evaluate(params, valid_X, valid_y, shakespeare_config)
        accs_all.append(float(acc))
        accs_steps.append(global_step - 1)

        perplexity = ep_perplexity(params, valid_X, valid_y, shakespeare_config)
        perplexities_all.append(float(perplexity))
        perplexities_steps.append(global_step - 1)

        key, gen_key = jr.split(key)
        generated_text = generate_text(
            params,
            seed_context,
            idx_to_char,
            shakespeare_config,
            num_chars=args.gen_chars,
            temperature=args.temperature,
            key=gen_key,
        )
        generated_texts_all.append(generated_text)
        generated_texts_epochs.append(epoch)

        seed_text = "".join([idx_to_char[int(idx)] for idx in seed_context])

        print(
            f"epoch {epoch:5d} | "
            f"step {global_step:6d}/{shakespeare_config.max_steps} | "
            f"train loss {mean_train_loss:8.4f} | "
            f"valid acc {float(acc):6.4f} | "
            f"perplexity {perplexity:8.4f} | "
            f"force {eval_force:.6f} | "
            f"{epoch_time:3.3f}s / epoch"
        )
        print(f"  Seed: '{seed_text}'")
        print(f"  Generated: '{generated_text}'")

        wandb.log(
            {
                "valid/accuracy": float(acc),
                "valid/perplexity": float(perplexity),
                "valid/force_mag": eval_force,
                "valid/eval_epoch": epoch,
                "generated_text": wandb.Html(
                    f"<pre>Seed: {seed_text}\nGenerated: {generated_text}</pre>"
                ),
            },
            step=global_step - 1,
        )

        save_params(params, f"{output_dir}/model_shakespeare.npz")
        if acc >= 0.5:
            save_params(params, f"{output_dir}/model_shakespeare_best.npz")

        epoch += 1

    save_metrics(
        {"step": losses_steps, "loss": losses_all},
        f"{output_dir}/losses_shakespeare.json",
    )
    save_metrics(
        {"step": accs_steps, "accuracy": accs_all},
        f"{output_dir}/accs_shakespeare.json",
    )
    save_metrics(
        {"step": perplexities_steps, "perplexity": perplexities_all},
        f"{output_dir}/perplexities_shakespeare.json",
    )
    save_metrics(
        {"epoch": generated_texts_epochs, "text": generated_texts_all},
        f"{output_dir}/generated_texts_shakespeare.json",
    )

    save_params(params, f"{output_dir}/model_shakespeare.npz")
    print(
        f"Training complete. Model saved to '{output_dir}/model_shakespeare.npz'."
    )

    print("Creating training plots...")
    plot_path = plot_training_metrics(
        losses_steps,
        losses_all,
        accs_steps,
        accs_all,
        perplexities_steps,
        perplexities_all,
        output_path=f"{output_dir}/training_metrics_shakespeare.png",
    )
    print(f"Plots saved to {plot_path}")
    wandb.log({"training_summary": wandb.Image(plot_path)})

    wandb.finish()
