"""Generate configs/config_XX.json files from hyperparameter sweep definitions.

Sweep params are either:
  - choice([...])    — sample uniformly from a discrete list
  - loguniform(a, b) — sample from log-uniform distribution over [a, b]
  - uniform(a, b)    — sample from uniform distribution over [a, b]
"""

import json
import os
from dataclasses import dataclass

import numpy as np

# ============================================================
# Sweep primitives
# ============================================================


@dataclass
class choice:
    values: list


@dataclass
class loguniform:
    low: float
    high: float


@dataclass
class uniform:
    low: float
    high: float


# ============================================================
# Define sweeps here.
# ============================================================

SWEEP = {
    # Discrete / architectural params
    "D": choice([32]),
    "M": choice([16]),
    "batch_size": choice([128]),
    # "lr_end_factor": choice([0.003, 0.03, 0.3, 1]),
    # Continuous params — log-uniform over range
    "lr_peak_value": loguniform(1e-4, 3e-2),
    "lr_end_factor": loguniform(1e-3, 1e-0),
    # "lr_peak_value": choice([0.01]),
    # "xi_attn_embed_raw_scale": loguniform(1e-3, 2e-2),
    "xi_attn_embed_raw_scale": choice([1e-3]),
    # "xi_hopf_raw_scale": loguniform(1e-3, 2e-2),
    "xi_hopf_raw_scale": choice([0.06]),
}

NUM_CONFIGS = 50
SEED = 42

# ============================================================
# Defaults for all non-swept parameters.
# ============================================================

DEFAULTS = {
    "L": 16,
    "xi_attn_embed_raw_scale": 0.1,
    "xi_hopf_raw_scale": 0.06,
    "step_size": 0.001,
    "T_final": 1.0,
    "train_epochs": 10,
    "max_steps": 20_000,
    "seed": 0,
    "tau_v": 0.1,
    "tau_h": 0.01,
    "lr_init_value": 0.0,
    "lr_peak_value": 0.01,
    "beta": 0.1,
    "max_norm": 1.0,
    "slow_weight_decay": 5e-05,
    "fast_weight_decay": 0.0,
    "force_penalty_start": 999_999_999,
    "force_penalty_duration": 1,
    "force_penalty_scale": 0.0,
}

# ============================================================


def sample_param(spec, rng):
    if isinstance(spec, choice):
        return rng.choice(spec.values)
    elif isinstance(spec, loguniform):
        log_low, log_high = np.log(spec.low), np.log(spec.high)
        return float(np.exp(rng.uniform(log_low, log_high)))
    elif isinstance(spec, uniform):
        return float(rng.uniform(spec.low, spec.high))
    else:
        return spec


def generate():
    rng = np.random.default_rng(SEED)
    sweep_keys = list(SWEEP.keys())

    print(f"Sampling {NUM_CONFIGS} configs over {len(sweep_keys)} params: {sweep_keys}")

    # Clear old configs
    os.makedirs("configs", exist_ok=True)
    for old in os.listdir("configs"):
        if old.startswith("config_") and old.endswith(".json"):
            os.remove(f"configs/{old}")

    for i in range(1, NUM_CONFIGS + 1):
        cfg = dict(DEFAULTS)
        for key in sweep_keys:
            val = sample_param(SWEEP[key], rng)
            cfg[key] = int(val) if isinstance(val, (int, np.integer)) else float(val)

        path = f"configs/config_{i:03d}.json"
        with open(path, "w") as f:
            json.dump(cfg, f, indent=2)
            f.write("\n")

    print(f"Wrote configs/config_001.json .. configs/config_{NUM_CONFIGS:03d}.json")


if __name__ == "__main__":
    generate()