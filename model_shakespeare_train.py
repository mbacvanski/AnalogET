import argparse
from datetime import datetime
import json
import os
from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import time

import jax
import jax.numpy as jnp
import jax.random as jr
import optax

import wandb

from config import Config
from data import load_shakespeare_dataset, prepare_shakespeare_dataset
from model import evaluate, force_penalty_weight, init_params, label_tree, loss_fn
from utils import calculate_perplexity, generate_text, load_params, plot_training_metrics, save_metrics, save_params


def get_T_final_for_epoch(epoch: int, base_T_final: float = 0.01, phase_length: int = 20) -> float:
    """
    Calculate T_final based on epoch number.
    
    T_final increases by base_T_final every phase_length epochs:
    - Epochs 0-19: T_final = 0.01 (n_steps = 10 with step_size=0.001)
    - Epochs 20-39: T_final = 0.02 (n_steps = 20)
    - Epochs 40-59: T_final = 0.03 (n_steps = 30)
    - ...
    """
    phase = epoch // phase_length
    return base_T_final * (phase + 1)


def params_weight_stats(params) -> Dict[str, float]:
    """
    Compute lightweight global summary stats over all floating-point parameter arrays.

    Returns host scalars suitable for `wandb.log()`.
    """
    leaves = jax.tree_util.tree_leaves(params)

    count_i = jnp.array(0, dtype=jnp.int64)
    finite_i = jnp.array(0, dtype=jnp.int64)
    sum_f = jnp.array(0.0, dtype=jnp.float32)
    sumsq_f = jnp.array(0.0, dtype=jnp.float32)
    abs_sum_f = jnp.array(0.0, dtype=jnp.float32)
    min_f = jnp.array(jnp.inf, dtype=jnp.float32)
    max_f = jnp.array(-jnp.inf, dtype=jnp.float32)
    abs_max_f = jnp.array(0.0, dtype=jnp.float32)

    saw_any = False
    for leaf in leaves:
        if not hasattr(leaf, "dtype") or not jnp.issubdtype(leaf.dtype, jnp.floating):
            continue
        saw_any = True
        x = leaf.astype(jnp.float32)
        count_i = count_i + x.size
        sum_f = sum_f + jnp.sum(x)
        sumsq_f = sumsq_f + jnp.sum(x * x)
        abs_sum_f = abs_sum_f + jnp.sum(jnp.abs(x))
        min_f = jnp.minimum(min_f, jnp.min(x))
        max_f = jnp.maximum(max_f, jnp.max(x))
        abs_max_f = jnp.maximum(abs_max_f, jnp.max(jnp.abs(x)))
        finite_i = finite_i + jnp.sum(jnp.isfinite(x))

    if not saw_any:
        return {}

    count_f = count_i.astype(jnp.float32)
    mean_f = sum_f / count_f
    var_f = jnp.maximum(sumsq_f / count_f - mean_f * mean_f, 0.0)
    std_f = jnp.sqrt(var_f)
    rms_f = jnp.sqrt(sumsq_f / count_f)
    l2_f = jnp.sqrt(sumsq_f)
    abs_mean_f = abs_sum_f / count_f
    finite_frac_f = finite_i.astype(jnp.float32) / count_f

    out = {
        "weights/num_params": int(jax.device_get(count_i)),
        "weights/finite_frac": float(jax.device_get(finite_frac_f)),
        "weights/mean": float(jax.device_get(mean_f)),
        "weights/std": float(jax.device_get(std_f)),
        "weights/rms": float(jax.device_get(rms_f)),
        "weights/l2": float(jax.device_get(l2_f)),
        "weights/min": float(jax.device_get(min_f)),
        "weights/max": float(jax.device_get(max_f)),
        "weights/abs_mean": float(jax.device_get(abs_mean_f)),
        "weights/abs_max": float(jax.device_get(abs_max_f)),
    }
    return out


def _tree_path_to_str(path: tuple[Any, ...]) -> str:
    parts = []
    for entry in path:
        if hasattr(entry, "key"):
            parts.append(str(entry.key))
        elif hasattr(entry, "name"):
            parts.append(str(entry.name))
        elif hasattr(entry, "idx"):
            parts.append(str(entry.idx))
        else:
            parts.append(str(entry))
    return "/".join(parts) if parts else "root"


def params_weight_snapshot(params) -> Dict[str, Any]:
    """
    Build per-parameter histogram snapshots for W&B.

    Keys are path-based for stable grouping across runs.
    """
    snapshot: Dict[str, Any] = {}
    leaves_with_path = jax.tree_util.tree_leaves_with_path(params)
    for path, leaf in leaves_with_path:
        if not hasattr(leaf, "dtype") or not jnp.issubdtype(leaf.dtype, jnp.floating):
            continue
        values = jnp.ravel(leaf.astype(jnp.float32))
        finite_values = values[jnp.isfinite(values)]
        if finite_values.size == 0:
            continue
        values_np = jax.device_get(finite_values)
        name = _tree_path_to_str(path)
        snapshot[f"weights_snapshot/{name}"] = wandb.Histogram(values_np)
    return snapshot


def save_and_log_epoch_checkpoint(params, output_dir: str, epoch: int) -> str:
    """Save full checkpoint tensors and upload them to W&B run files."""
    checkpoints_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    checkpoint_path = os.path.join(
        checkpoints_dir, f"model_shakespeare_epoch_{epoch:04d}.npz"
    )
    save_params(params, checkpoint_path)
    wandb.save(checkpoint_path, base_path=output_dir, policy="now")
    return checkpoint_path


def parse_wandb_run_ref(run_ref: str) -> str:
    """
    Normalize a run reference to `entity/project/run_id`.

    Accepted forms:
    - entity/project/run_id
    - entity/project/runs/run_id
    - https://wandb.ai/entity/project/runs/run_id
    """
    ref = run_ref.strip()
    if not ref:
        raise ValueError("W&B run reference is empty.")

    if ref.startswith("wandb.ai/") or ref.startswith("www.wandb.ai/"):
        ref = f"https://{ref}"

    if ref.startswith("http://") or ref.startswith("https://"):
        parsed = urlparse(ref)
        parts = [p for p in parsed.path.split("/") if p]
    else:
        parts = [p for p in ref.split("/") if p]

    if len(parts) == 4 and parts[2] == "runs":
        return f"{parts[0]}/{parts[1]}/{parts[3]}"
    if len(parts) == 3:
        return f"{parts[0]}/{parts[1]}/{parts[2]}"

    raise ValueError(
        "Could not parse W&B run reference. Expected "
        "`entity/project/run_id` or `.../entity/project/runs/run_id`."
    )


def download_wandb_checkpoint(
    run_path: str, file_name: str, download_root: str
) -> Tuple[str, Any]:
    """Download `file_name` from a W&B run and return local path + run object."""
    api = wandb.Api()
    run = api.run(run_path)
    run_file = run.file(file_name)
    if run_file is None:
        available_npz = sorted(
            f.name for f in run.files(per_page=1000) if f.name.endswith(".npz")
        )
        available_note = (
            f"Available .npz files: {', '.join(available_npz)}"
            if available_npz
            else "No .npz files were found in this run."
        )
        raise FileNotFoundError(
            f"File '{file_name}' not found in W&B run '{run_path}'. {available_note} "
            "If this run only logged `weights_snapshot/*` histograms, those are not "
            "recoverable as exact model tensors for resume. Also note: "
            "`run-<id>-history` / `wandb-history` artifacts contain metric history, "
            "not model checkpoint tensors."
        )

    local_root = os.path.join(download_root, *run_path.split("/"))
    os.makedirs(local_root, exist_ok=True)

    downloaded = run_file.download(root=local_root, replace=False, exist_ok=True)
    local_path = os.path.abspath(downloaded.name)
    downloaded.close()
    return local_path, run


def _as_int(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError(f"Boolean value is not valid here: {value}")
    return int(value)


def _wandb_cfg_value(value: Any) -> Any:
    if isinstance(value, dict) and "value" in value:
        return value["value"]
    return value


def validate_wandb_run_config_compat(run: Any, cfg: Config) -> None:
    """Fail fast if W&B run config dimensions disagree with active local config."""
    run_cfg = getattr(run, "config", {}) or {}
    expected = {
        "D": int(cfg.D),
        "M": int(cfg.M),
        "L": int(cfg.L),
        "vocab_size": int(cfg.vocab_size),
    }

    mismatches: List[str] = []
    for key, expected_value in expected.items():
        if key not in run_cfg:
            continue
        observed_raw = _wandb_cfg_value(run_cfg[key])
        try:
            observed = _as_int(observed_raw)
        except (TypeError, ValueError):
            mismatches.append(
                f"{key}: run has non-integer value {observed_raw!r}, expected {expected_value}"
            )
            continue
        if observed != expected_value:
            mismatches.append(
                f"{key}: run has {observed}, expected {expected_value}"
            )

    if mismatches:
        mismatch_text = "\n  - ".join(mismatches)
        raise ValueError(
            "W&B run config is incompatible with the active local config:\n"
            f"  - {mismatch_text}"
        )


def validate_checkpoint_shapes(params: Dict[str, Any], cfg: Config) -> None:
    """Validate checkpoint tensor shapes against active config dimensions."""
    expected_shapes = {
        "xi_attn_embed_raw": (cfg.vocab_size, cfg.D),
        "xi_pos_raw": (cfg.L, cfg.D),  # optional for old checkpoints
        "xi_hopf_raw": (cfg.M, cfg.D),
        "b": (cfg.L,),
        "c": (cfg.M,),
        "a": (cfg.D,),
        "W_dec": (cfg.vocab_size, cfg.D),
        "b_dec": (cfg.vocab_size,),
    }
    optional_keys = {"xi_pos_raw"}

    missing: List[str] = []
    mismatched: List[str] = []
    for key, shape in expected_shapes.items():
        if key not in params:
            if key in optional_keys:
                continue
            missing.append(key)
            continue
        observed_shape = tuple(int(dim) for dim in params[key].shape)
        if observed_shape != tuple(int(dim) for dim in shape):
            mismatched.append(
                f"{key}: checkpoint has {observed_shape}, expected {shape}"
            )

    if missing or mismatched:
        details: List[str] = []
        if missing:
            details.append(f"missing keys: {', '.join(sorted(missing))}")
        if mismatched:
            details.append("shape mismatches:\n  - " + "\n  - ".join(mismatched))
        raise ValueError("Checkpoint is incompatible with active config:\n" + "\n".join(details))


def build_epoch_checkpoint_file(epoch: int) -> str:
    if epoch < 0:
        raise ValueError(f"Checkpoint epoch must be >= 0, got {epoch}")
    return f"checkpoints/model_shakespeare_epoch_{epoch:04d}.npz"


def infer_ctx_length_from_wandb_run(run: Any) -> int:
    run_cfg = getattr(run, "config", {}) or {}
    for key in ("ctx_length", "L"):
        if key not in run_cfg:
            continue
        value = _wandb_cfg_value(run_cfg[key])
        try:
            ctx = _as_int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid {key} in W&B run config: {value!r}") from exc
        if ctx <= 0:
            raise ValueError(f"Invalid {key} in W&B run config: {ctx} (must be > 0)")
        return ctx
    raise ValueError(
        "Could not infer ctx_length from W&B run config; expected `ctx_length` or `L`."
    )


def extract_wandb_config_init_kwargs(run: Any) -> Dict[str, Any]:
    """Extract Config __init__ kwargs from a W&B run config."""
    run_cfg = getattr(run, "config", {}) or {}
    keys = [
        "D",
        "M",
        "beta",
        "xi_attn_embed_raw_scale",
        "xi_pos_raw_scale",
        "xi_hopf_raw_scale",
        "step_size",
        "T_final",
        "batch_size",
        "train_epochs",
        "max_steps",
        "seed",
        "tau_v",
        "tau_h",
        "use_multi_transform",
        "lr_init_value",
        "lr_peak_value",
        "learning_rate",
        "lr_end_factor",
        "max_norm",
        "slow_weight_decay",
        "fast_weight_decay",
        "force_penalty_start",
        "force_penalty_duration",
        "force_penalty_scale",
    ]
    int_keys = {
        "D",
        "M",
        "batch_size",
        "train_epochs",
        "max_steps",
        "seed",
        "force_penalty_start",
        "force_penalty_duration",
    }
    float_keys = {
        "beta",
        "xi_attn_embed_raw_scale",
        "xi_pos_raw_scale",
        "xi_hopf_raw_scale",
        "step_size",
        "T_final",
        "tau_v",
        "tau_h",
        "lr_init_value",
        "lr_peak_value",
        "learning_rate",
        "lr_end_factor",
        "max_norm",
        "slow_weight_decay",
        "fast_weight_decay",
        "force_penalty_scale",
    }
    bool_keys = {"use_multi_transform"}

    out: Dict[str, Any] = {}
    for key in keys:
        if key not in run_cfg:
            continue
        raw = _wandb_cfg_value(run_cfg[key])
        try:
            if key in int_keys:
                out[key] = int(raw)
            elif key in float_keys:
                out[key] = float(raw)
            elif key in bool_keys:
                out[key] = bool(raw)
            else:
                out[key] = raw
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid `{key}` value in W&B run config: {raw!r}") from exc
    return out


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train Shakespeare character prediction model")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to JSON config file; if omitted with W&B init, build config from W&B run config",
    )
    parser.add_argument(
        "--ctx_length",
        type=int,
        default=None,
        help="Context length for training sequences (default: 16, or inferred from W&B run)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Temperature for text generation sampling (default: 0.8)",
    )
    parser.add_argument(
        "--gen_chars",
        type=int,
        default=32,
        help="Number of characters to generate at each epoch (default: 32)",
    )
    parser.add_argument(
        "--sample_mode",
        action="store_true",
        help="Use only 1000 batches of data for quick hyperparameter search",
    )
    parser.add_argument(
        "--init_weights",
        type=str,
        default=None,
        help="Path to .npz file to initialize model weights from (e.g., data/shakespeare/20260208_100248/model_shakespeare.npz)",
    )
    parser.add_argument(
        "--init_wandb_run",
        type=str,
        default=None,
        help="W&B source run reference for initialization (`entity/project/run_id` or run URL)",
    )
    parser.add_argument(
        "--init_wandb_run_id",
        type=str,
        default=None,
        help="W&B source run ID only (e.g., k9arebfp); requires --init_wandb_epoch",
    )
    parser.add_argument(
        "--init_wandb_epoch",
        type=int,
        default=None,
        help="Epoch to resume from when using --init_wandb_run_id or --init_wandb_run",
    )
    parser.add_argument(
        "--init_wandb_entity",
        type=str,
        default="qpaig",
        help="W&B entity for --init_wandb_run_id mode (default: qpaig)",
    )
    parser.add_argument(
        "--init_wandb_project",
        type=str,
        default="analog-et",
        help="W&B project for --init_wandb_run_id mode (default: analog-et)",
    )
    parser.add_argument(
        "--init_wandb_file",
        type=str,
        default=None,
        help="Checkpoint file name within the W&B run (default: model_shakespeare.npz, or derived from --init_wandb_epoch)",
    )
    parser.add_argument(
        "--init_wandb_root",
        type=str,
        default="data/wandb_downloads",
        help="Local directory root for downloaded W&B checkpoints",
    )
    parser.add_argument(
        "--curriculum_T_final",
        action="store_true",
        help="Enable curriculum learning for T_final: increase T_final by 0.01 every 20 epochs while keeping step_size=0.001",
    )
    parser.add_argument(
        "--curriculum_base_T_final",
        type=float,
        default=0.01,
        help="Base T_final for curriculum learning (default: 0.01)",
    )
    parser.add_argument(
        "--curriculum_phase_length",
        type=int,
        default=20,
        help="Number of epochs per phase in curriculum learning (default: 20)",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=512,
        help="Batch size for validation accuracy/perplexity to bound memory (default: 512)",
    )
    args = parser.parse_args()

    using_wandb_source = bool(args.init_wandb_run or args.init_wandb_run_id)
    if args.init_weights and using_wandb_source:
        parser.error(
            "Use only one of --init_weights or W&B init args "
            "(--init_wandb_run/--init_wandb_run_id)."
        )
    if args.init_wandb_run and args.init_wandb_run_id:
        parser.error("Use only one of --init_wandb_run or --init_wandb_run_id.")
    if args.init_wandb_run_id and args.init_wandb_epoch is None:
        parser.error("--init_wandb_run_id requires --init_wandb_epoch.")
    if args.init_wandb_epoch is not None and not using_wandb_source:
        parser.error("--init_wandb_epoch requires --init_wandb_run or --init_wandb_run_id.")
    if args.init_wandb_epoch is not None and args.init_wandb_file is not None:
        parser.error("Use either --init_wandb_epoch or --init_wandb_file, not both.")
    if args.init_wandb_epoch is not None and args.init_wandb_epoch < 0:
        parser.error(f"--init_wandb_epoch must be >= 0, got {args.init_wandb_epoch}.")
    
    effective_init_weights = args.init_weights
    resolved_wandb_run = None
    resolved_wandb_file = None
    resolved_wandb_epoch = args.init_wandb_epoch
    source_run = None
    wandb_config_kwargs = None
    if args.init_wandb_run or args.init_wandb_run_id:
        if args.init_wandb_run_id:
            resolved_wandb_run = (
                f"{args.init_wandb_entity}/{args.init_wandb_project}/{args.init_wandb_run_id}"
            )
        else:
            resolved_wandb_run = parse_wandb_run_ref(args.init_wandb_run)

        if resolved_wandb_epoch is not None:
            resolved_wandb_file = build_epoch_checkpoint_file(resolved_wandb_epoch)
        elif args.init_wandb_file:
            resolved_wandb_file = args.init_wandb_file
        else:
            resolved_wandb_file = "model_shakespeare.npz"

        print(
            f"Downloading initialization checkpoint '{resolved_wandb_file}' from W&B run "
            f"'{resolved_wandb_run}'..."
        )
        effective_init_weights, source_run = download_wandb_checkpoint(
            run_path=resolved_wandb_run,
            file_name=resolved_wandb_file,
            download_root=args.init_wandb_root,
        )
        print(f"Downloaded checkpoint to: {effective_init_weights}")
        if args.ctx_length is None:
            ctx_length = infer_ctx_length_from_wandb_run(source_run)
            print(f"Inferred ctx_length={ctx_length} from W&B run config.")
        else:
            ctx_length = args.ctx_length
        if args.config is None:
            wandb_config_kwargs = extract_wandb_config_init_kwargs(source_run)
    else:
        if args.ctx_length is not None:
            ctx_length = args.ctx_length
        elif args.config:
            with open(args.config, "r") as f:
                pre_cfg = json.load(f)
            ctx_length = int(pre_cfg.get("L", 16))
            print(f"Inferred ctx_length={ctx_length} from local config file.")
        else:
            ctx_length = 16

    # Load or prepare Shakespeare dataset
    filename_prefix = "shakespeare_data"
    dataset_exists = os.path.exists(f"data/{filename_prefix}_ctx{ctx_length}_train_X.txt")

    if dataset_exists:
        print(f"Loading Shakespeare dataset from saved files (ctx_length={ctx_length})...")
        train_X, train_y, valid_X, valid_y, char_to_idx, idx_to_char = load_shakespeare_dataset(
            ctx_length=ctx_length, filename_prefix=filename_prefix
        )
    else:
        print(f"Preparing Shakespeare dataset (ctx_length={ctx_length})...")
        train_X, train_y, valid_X, valid_y, char_to_idx, idx_to_char = prepare_shakespeare_dataset(
            ctx_length=ctx_length, train_ratio=0.9, filename_prefix=filename_prefix
        )

    vocab_size = len(char_to_idx)
    print(f"Using vocab_size={vocab_size}, ctx_length={ctx_length}")
    
    # Apply sample mode if requested (limit to 1000 batches worth of data)
    if args.sample_mode:
        max_samples = 1000 * 256  # 1000 batches * batch_size
        if len(train_X) > max_samples:
            print(f"Sample mode: Using {max_samples} training samples (1000 batches)")
            train_X = train_X[:max_samples]
            train_y = train_y[:max_samples]
        if len(valid_X) > max_samples:
            print(f"Sample mode: Using {max_samples} validation samples")
            valid_X = valid_X[:max_samples]
            valid_y = valid_y[:max_samples]

    # Initialize config from JSON file, W&B run config, or defaults.
    if args.config:
        print(f"Loading config from {args.config}")
        with open(args.config, "r") as f:
            config_dict = json.load(f)
        shakespeare_config = Config(
            L=ctx_length,
            vocab_size=vocab_size,
            **{k: v for k, v in config_dict.items() if k not in ["L", "vocab_size"]}
        )
        if "L" in config_dict and config_dict["L"] != ctx_length:
            raise ValueError(f"Config file has L={config_dict['L']}, but overriding with ctx_length={ctx_length}")
    elif wandb_config_kwargs is not None:
        print("Building config from W&B run config (no local --config provided).")
        shakespeare_config = Config(
            L=ctx_length,
            vocab_size=vocab_size,
            **wandb_config_kwargs,
        )
    else:
        shakespeare_config = Config(
            L=ctx_length,
            vocab_size=vocab_size,
        )

    if source_run is not None:
        validate_wandb_run_config_compat(source_run, shakespeare_config)

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"data/shakespeare/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Save config to output directory for reproducibility
    config_save_dict = {
        k: getattr(shakespeare_config, k) 
        for k in dir(shakespeare_config)
        if not k.startswith("_") and not callable(getattr(shakespeare_config, k))
    }
    config_save_dict['ctx_length'] = ctx_length
    config_save_dict['sample_mode'] = args.sample_mode
    config_save_dict['log_full_weights_each_epoch'] = True
    config_save_dict['eval_batch_size'] = args.eval_batch_size
    config_save_dict['temperature'] = args.temperature
    config_save_dict['gen_chars'] = args.gen_chars
    if args.config:
        config_save_dict['config_file'] = args.config
    elif source_run is not None:
        config_save_dict['config_from_wandb_run'] = True
    if effective_init_weights:
        config_save_dict['init_weights'] = effective_init_weights
    if resolved_wandb_run:
        config_save_dict['init_wandb_run'] = resolved_wandb_run
        config_save_dict['init_wandb_file'] = resolved_wandb_file
        config_save_dict['init_wandb_root'] = args.init_wandb_root
        if args.init_wandb_run_id:
            config_save_dict['init_wandb_run_id'] = args.init_wandb_run_id
            config_save_dict['init_wandb_entity'] = args.init_wandb_entity
            config_save_dict['init_wandb_project'] = args.init_wandb_project
        if resolved_wandb_epoch is not None:
            config_save_dict['init_wandb_epoch'] = resolved_wandb_epoch
    config_save_dict['curriculum_T_final'] = args.curriculum_T_final
    if args.curriculum_T_final:
        config_save_dict['curriculum_base_T_final'] = args.curriculum_base_T_final
        config_save_dict['curriculum_phase_length'] = args.curriculum_phase_length
    
    with open(f"{output_dir}/run_config.json", "w") as f:
        json.dump(config_save_dict, f, indent=2)
    print(f"Run configuration saved to {output_dir}/run_config.json")
    
    # ---- W&B init ----
    # Try to serialize Config cleanly (works for dataclass / simple objects)
    try:
        config_dict = vars(shakespeare_config)
    except TypeError:
        config_dict = {k: getattr(shakespeare_config, k) for k in dir(shakespeare_config)
                       if not k.startswith("_") and isinstance(getattr(shakespeare_config, k), (int, float, str, bool))}

    run = wandb.init(
        entity='qpaig',
        project="analog-et",
        name=f"ctx{shakespeare_config.L}_bs{shakespeare_config.batch_size}",
        config=config_dict,
    )
    # -------------------------

    key = jr.PRNGKey(shakespeare_config.seed)

    # Initialize params - either from file or fresh
    if effective_init_weights:
        print(f"Loading initial weights from {effective_init_weights}")
        params = load_params(effective_init_weights)
        validate_checkpoint_shapes(params, shakespeare_config)
        print(f"Loaded {len(params)} parameter arrays from checkpoint")
    else:
        params = init_params(key, shakespeare_config)

    # Log initial parameter stats once (helps catch NaNs early)
    init_weight_stats = params_weight_stats(params)
    if init_weight_stats:
        wandb.log({**init_weight_stats, "epoch": 0}, step=0)

    num_train = train_X.shape[0]
    num_batches = (num_train + shakespeare_config.batch_size - 1) // shakespeare_config.batch_size
    total_steps = shakespeare_config.train_epochs * num_batches


    def lr_sched(
        peak: float, warmup_steps: int = 0
    ) -> optax.Schedule:
        """warm up to peak, then decay to peak * end_factor"""
        return optax.warmup_cosine_decay_schedule(
            init_value=shakespeare_config.lr_init_value,
            peak_value=peak,
            warmup_steps=warmup_steps,
            decay_steps=max(1, total_steps - warmup_steps),
            end_value=peak * shakespeare_config.lr_end_factor,
        )


    # ---- create optimizer based on config ----
    if shakespeare_config.use_multi_transform:
        # Multi-transform optimizer with fast/slow parameters
        lr_schedule_fast = lr_sched(shakespeare_config.lr_peak_value)
        lr_schedule_slow = lr_sched(shakespeare_config.lr_peak_value)
        
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
        
        optimizer = optax.multi_transform(
            {"fast": tx_fast, "slow": tx_slow}, label_tree(params)
        )
    else:
        optimizer = optax.chain(
            optax.clip_by_global_norm(shakespeare_config.max_norm),
            optax.adamw(
                learning_rate=shakespeare_config.learning_rate,
            ),
        )
    # -------------------------------------------------------
    
    opt_state = optimizer.init(params)


    # Function to create a train_step with a specific config captured in closure
    def make_train_step(cfg):
        @jax.jit
        def train_step(params, opt_state, train_X, train_Y, force_weight):
            loss, grads = jax.value_and_grad(loss_fn)(
                params, train_X, train_Y, force_weight, cfg
            )
            grad_norm = optax.global_norm(grads)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss, grad_norm
        return train_step


    # Initialize the train_step with current config
    train_step = make_train_step(shakespeare_config)
    current_T_final = shakespeare_config.T_final
    current_config = shakespeare_config


    losses_all = []
    losses_steps: List[int] = []
    accs_all = []
    accs_steps: List[int] = []
    perplexities_all = []
    perplexities_steps: List[int] = []
    generated_texts_all = []
    generated_texts_epochs: List[int] = []

    global_step = 0
    
    # Prepare seed contexts for text generation (one from train, one from validation)
    seed_context_train = train_X[0]
    seed_context_valid = valid_X[0]

    for epoch in range(shakespeare_config.train_epochs):
        # Handle curriculum learning for T_final
        if args.curriculum_T_final:
            new_T_final = get_T_final_for_epoch(
                epoch, 
                base_T_final=args.curriculum_base_T_final,
                phase_length=args.curriculum_phase_length
            )
            
            if new_T_final != current_T_final:
                # T_final changed, create new config and re-JIT train_step
                print(f"\n=== Curriculum Update at epoch {epoch}: T_final {current_T_final:.4f} -> {new_T_final:.4f} (n_steps: {int(new_T_final / shakespeare_config.step_size)}) ===\n")
                
                # Create new config with updated T_final but same step_size
                current_config = Config(
                    L=shakespeare_config.L,
                    vocab_size=shakespeare_config.vocab_size,
                    D=shakespeare_config.D,
                    M=shakespeare_config.M,
                    beta=shakespeare_config.beta,
                    xi_attn_embed_raw_scale=shakespeare_config.xi_attn_embed_raw_scale,
                    xi_pos_raw_scale=shakespeare_config.xi_pos_raw_scale,
                    xi_hopf_raw_scale=shakespeare_config.xi_hopf_raw_scale,
                    step_size=shakespeare_config.step_size,  # Keep step_size constant
                    T_final=new_T_final,  # Update T_final
                    batch_size=shakespeare_config.batch_size,
                    train_epochs=shakespeare_config.train_epochs,
                    seed=shakespeare_config.seed,
                    tau_v=shakespeare_config.tau_v,
                    tau_h=shakespeare_config.tau_h,
                    use_multi_transform=shakespeare_config.use_multi_transform,
                    lr_init_value=shakespeare_config.lr_init_value,
                    lr_peak_value=shakespeare_config.lr_peak_value,
                    learning_rate=shakespeare_config.learning_rate,
                    lr_end_factor=shakespeare_config.lr_end_factor,
                    max_norm=shakespeare_config.max_norm,
                    slow_weight_decay=shakespeare_config.slow_weight_decay,
                    fast_weight_decay=shakespeare_config.fast_weight_decay,
                    force_penalty_start=shakespeare_config.force_penalty_start,
                    force_penalty_duration=shakespeare_config.force_penalty_duration,
                    force_penalty_scale=shakespeare_config.force_penalty_scale,
                )
                
                # Re-create the train_step with new config
                train_step = make_train_step(current_config)
                current_T_final = new_T_final
                
                # Log to W&B
                wandb.log({
                    "curriculum/T_final": new_T_final,
                    "curriculum/n_steps": current_config.n_steps,
                    "epoch": epoch,
                }, step=global_step)
        t_start = time.time()
        key, key_perm = jr.split(key)
        index_perm = jr.permutation(key_perm, num_train)
        train_X_epoch = train_X[index_perm]
        train_y_epoch = train_y[index_perm]
        losses_epoch = []

        lam_force = jnp.asarray(force_penalty_weight(epoch, current_config), dtype=jnp.float32)

        for batch in range(num_batches):
            start = batch * current_config.batch_size
            stop = min((batch + 1) * current_config.batch_size, num_train)
            key, sub = jr.split(key)
            batch_train_X = train_X_epoch[start:stop]
            batch_train_y = train_y_epoch[start:stop]

            params, opt_state, loss, grad_norm = train_step(
                params,
                opt_state,
                batch_train_X,
                batch_train_y,
                force_weight=lam_force,
            )

            losses_epoch.append(loss)
            losses_all.append(float(loss))
            losses_steps.append(global_step)

            # ---- W&B per-step logging ----
            # If logging every step is too chatty, gate with: if global_step % 10 == 0:
            log_dict = {
                "train/loss": float(loss),
                "train/force_w": float(lam_force),
                "train/grad_norm": float(grad_norm),
                "epoch": epoch,
            }
            
            # Add learning rate logs based on optimizer type
            if shakespeare_config.use_multi_transform:
                log_dict["train/lr_fast"] = float(lr_schedule_fast(global_step))
                log_dict["train/lr_slow"] = float(lr_schedule_slow(global_step))
            else:
                log_dict["train/lr"] = float(shakespeare_config.learning_rate)
            
            wandb.log(log_dict, step=global_step)
            # ------------------------------------

            global_step += 1

        t_end = time.time()
        epoch_time = t_end - t_start
        mean_train_loss = float(jnp.mean(jnp.array(losses_epoch)))

        # Optional: log epoch aggregates every epoch (cheap and useful)
        wandb.log(
            {
                "epoch/train_loss_mean": mean_train_loss,
                "epoch/time_sec": float(epoch_time),
                "epoch": epoch,
            },
            step=global_step - 1,
        )

        # Log weight summary stats once per epoch (after updates)
        weight_stats = params_weight_stats(params)
        if weight_stats:
            wandb.log({**weight_stats, "epoch": epoch}, step=global_step - 1)

        # Log full weight histogram snapshot each epoch
        weight_snapshot = params_weight_snapshot(params)
        if weight_snapshot:
            wandb.log({**weight_snapshot, "epoch": epoch}, step=global_step - 1)

        if epoch % 1 == 0:
            acc = evaluate(params, valid_X, valid_y, current_config, batch_size=args.eval_batch_size)
            accs_all.append(float(acc))
            accs_steps.append((epoch + 1) * num_batches - 1)
            
            # Calculate perplexity
            perplexity = calculate_perplexity(
                params, valid_X, valid_y, current_config, batch_size=args.eval_batch_size
            )
            perplexities_all.append(float(perplexity))
            perplexities_steps.append((epoch + 1) * num_batches - 1)
            
            # Generate text from training example
            key, gen_key_train = jr.split(key)
            generated_text_train = generate_text(
                params,
                seed_context_train,
                idx_to_char,
                current_config,
                num_chars=args.gen_chars,
                temperature=args.temperature,
                key=gen_key_train,
            )
            
            # Generate text from validation example
            key, gen_key_valid = jr.split(key)
            generated_text_valid = generate_text(
                params,
                seed_context_valid,
                idx_to_char,
                current_config,
                num_chars=args.gen_chars,
                temperature=args.temperature,
                key=gen_key_valid,
            )
            
            # Store both generated texts
            generated_texts_all.append({
                "train": generated_text_train,
                "valid": generated_text_valid
            })
            generated_texts_epochs.append(epoch)
            
            # Show seed contexts for reference
            seed_text_train = "".join([idx_to_char[int(idx)] for idx in seed_context_train])
            seed_text_valid = "".join([idx_to_char[int(idx)] for idx in seed_context_valid])

            n_steps_info = f"n_steps {current_config.n_steps:3d} | " if args.curriculum_T_final else ""
            print(
                f"epoch {epoch:5d} | "
                f"{n_steps_info}"
                f"force_w {float(lam_force):5.3f} | "
                f"train loss {mean_train_loss:8.4f} | "
                f"valid acc {float(acc):6.4f} | "
                f"perplexity {perplexity:8.4f} | "
                f"{epoch_time:3.3f}s / epoch"
            )
            print(f"  Train Seed: '{seed_text_train}'")
            print(f"  Train Gen:  '{generated_text_train}'")
            print(f"  Valid Seed: '{seed_text_valid}'")
            print(f"  Valid Gen:  '{generated_text_valid}'")

            # ---- W&B eval logging ----
            eval_log_dict = {
                "valid/accuracy": float(acc),
                "valid/perplexity": float(perplexity),
                "valid/eval_epoch": epoch,
                "generated_text": wandb.Html(
                    f"<pre><b>Training Example:</b>\n"
                    f"Seed: {seed_text_train}\n"
                    f"Generated: {generated_text_train}\n\n"
                    f"<b>Validation Example:</b>\n"
                    f"Seed: {seed_text_valid}\n"
                    f"Generated: {generated_text_valid}</pre>"
                ),
            }
            if args.curriculum_T_final:
                eval_log_dict["curriculum/T_final"] = current_config.T_final
                eval_log_dict["curriculum/n_steps"] = current_config.n_steps
            wandb.log(eval_log_dict, step=global_step - 1)
            # --------------------------------

            epoch_checkpoint_path = save_and_log_epoch_checkpoint(params, output_dir, epoch)
            wandb.log(
                {"checkpoint/epoch_path": os.path.relpath(epoch_checkpoint_path, output_dir)},
                step=global_step - 1,
            )

            save_params(params, f"{output_dir}/model_shakespeare.npz")
            if acc >= 0.5:
                save_params(params, f"{output_dir}/model_shakespeare_best.npz")

    save_metrics({"step": losses_steps, "loss": losses_all}, f"{output_dir}/losses_shakespeare.json")
    save_metrics({"step": accs_steps, "accuracy": accs_all}, f"{output_dir}/accs_shakespeare.json")
    save_metrics({"step": perplexities_steps, "perplexity": perplexities_all}, f"{output_dir}/perplexities_shakespeare.json")
    save_metrics({"epoch": generated_texts_epochs, "text": generated_texts_all}, f"{output_dir}/generated_texts_shakespeare.json")

    save_params(params, f"{output_dir}/model_shakespeare.npz")
    print(f"Training complete. Model parameters saved to '{output_dir}/model_shakespeare.npz'.")
    
    # Create plots
    print("Creating training plots...")
    plot_path = plot_training_metrics(
        losses_steps, losses_all,
        accs_steps, accs_all,
        perplexities_steps, perplexities_all,
        output_path=f"{output_dir}/training_metrics_shakespeare.png"
    )
    print(f"Plots saved to {plot_path}")
    
    # Log final plot to W&B
    wandb.log({"training_summary": wandb.Image(plot_path)})

    wandb.finish()
