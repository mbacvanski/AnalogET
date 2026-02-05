import argparse
from dataclasses import replace
from datetime import datetime
import json
import os
from typing import List

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import time

import jax
import jax.numpy as jnp
import jax.random as jr
import optax

import wandb

from config import Config
from data import load_shakespeare_dataset, prepare_shakespeare_dataset
from model import (
    L_attn,
    L_hopf,
    evaluate,
    force_penalty_weight,
    get_xi_attn_embed,
    get_xi_hopf,
    get_xi_pos,
    init_params,
    label_tree,
    logits_from_v,
)
from utils import calculate_perplexity, generate_text, plot_training_metrics, save_metrics, save_params

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train Shakespeare character prediction model")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to JSON config file to initialize shakespeare_config",
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
        "--xi_pos_raw_scale",
        type=float,
        default=None,
        help="Scale for positional embedding init (xi_pos = square(raw)). If omitted, uses config/default.",
    )
    args = parser.parse_args()
    
    # Load or prepare Shakespeare dataset
    filename_prefix = "shakespeare_data"
    dataset_exists = os.path.exists(f"data/{filename_prefix}_train_X.txt")

    if dataset_exists:
        print("Loading Shakespeare dataset from saved files...")
        train_X, train_y, valid_X, valid_y, char_to_idx, idx_to_char = load_shakespeare_dataset(
            filename_prefix=filename_prefix
        )
    else:
        print("Preparing Shakespeare dataset...")
        train_X, train_y, valid_X, valid_y, char_to_idx, idx_to_char = prepare_shakespeare_dataset(
            ctx_length=16, train_ratio=0.9, filename_prefix=filename_prefix
        )

    vocab_size = len(char_to_idx)
    print(f"Using vocab_size={vocab_size}")
    
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

    # Initialize config from JSON file if provided, otherwise use defaults
    if args.config:
        print(f"Loading config from {args.config}")
        with open(args.config, "r") as f:
            config_dict = json.load(f)
        shakespeare_config = Config(
            L=config_dict.get("L", 16),
            vocab_size=vocab_size,
            **{k: v for k, v in config_dict.items() if k not in ["L", "vocab_size"]}
        )
    else:
        shakespeare_config = Config(
            L=16,
            vocab_size=vocab_size,
        )

    if args.xi_pos_raw_scale is not None:
        shakespeare_config = replace(shakespeare_config, xi_pos_raw_scale=float(args.xi_pos_raw_scale))

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
    config_save_dict['sample_mode'] = args.sample_mode
    config_save_dict['temperature'] = args.temperature
    config_save_dict['gen_chars'] = args.gen_chars
    if args.config:
        config_save_dict['config_file'] = args.config
    
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

    params = init_params(key, shakespeare_config)

    num_train = train_X.shape[0]
    num_batches = (num_train + shakespeare_config.batch_size - 1) // shakespeare_config.batch_size
    # total_steps = shakespeare_config.max_steps  # schedules removed; keep constant LR

    lr_fast = float(shakespeare_config.lr_peak_value)
    lr_slow = float(shakespeare_config.lr_peak_value)

    tx_fast = optax.chain(
        optax.clip_by_global_norm(shakespeare_config.max_norm),
        optax.adamw(
            learning_rate=lr_fast,
            weight_decay=shakespeare_config.fast_weight_decay,
        ),
    )
    tx_slow = optax.chain(
        optax.clip_by_global_norm(shakespeare_config.max_norm),
        optax.adamw(
            learning_rate=lr_slow,
            weight_decay=shakespeare_config.slow_weight_decay,
        ),
    )

    optimizer = optax.multi_transform(
        {"fast": tx_fast, "slow": tx_slow}, label_tree(params)
    )
    opt_state = optimizer.init(params)


    def infer_forward_euler_with_force_and_hidden(params, V0, ctx_bits):
        xi_attn_embed = get_xi_attn_embed(params)  # (vocab_size, D)
        L = ctx_bits.shape[1]
        xi_pos = get_xi_pos(params, L, V0.shape[1])  # (L, D)
        batch_xi_attn = xi_attn_embed[ctx_bits] + xi_pos[None, :, :]  # (B, L, D)
        xi_hopf = get_xi_hopf(params)  # (M, D)

        step_v = shakespeare_config.step_size / shakespeare_config.tau_v
        step_h = shakespeare_config.step_size / shakespeare_config.tau_h

        H_attn0 = jnp.einsum("bld,bd->bl", batch_xi_attn, V0) + params["b"]
        H_hopf0 = V0 @ xi_hopf.T + params["c"]

        def batch_energy(params, V, H_attn, H_hopf, F_attn, F_hopf):
            def _sample_energy(v, h_attn, h_hopf, f_attn, f_hopf, xi_attn):
                dv = v - params["a"]
                vis = 0.5 * jnp.dot(dv, dv)
                coupling = jnp.dot(v, xi_attn.T @ f_attn + xi_hopf.T @ f_hopf)
                att_bias = jnp.dot(f_attn, h_attn - params["b"])
                hopf_bias = jnp.dot(f_hopf, h_hopf - params["c"])
                return vis - coupling + att_bias + hopf_bias - L_attn(h_attn, shakespeare_config) - L_hopf(h_hopf)

            Eb = jax.vmap(_sample_energy, in_axes=(0, 0, 0, 0, 0, 0))(
                V, H_attn, H_hopf, F_attn, F_hopf, batch_xi_attn
            )
            return jnp.sum(Eb)

        grad_E = jax.grad(batch_energy, argnums=(1, 4, 5))  # grads wrt (V, F_attn, F_hopf)

        def grads_activation(V, H_attn, H_hopf):
            F_attn = jax.nn.softmax(shakespeare_config.beta * H_attn, axis=-1)
            F_hopf = jnp.maximum(H_hopf, 0.0)
            dE_dV, dE_dF_attn, dE_dF_hopf = grad_E(
                params, V, H_attn, H_hopf, F_attn, F_hopf
            )
            return dE_dV, dE_dF_attn, dE_dF_hopf

        def body(_, carry):
            V, H_attn, H_hopf = carry
            dE_dV, dE_dF_attn, dE_dF_hopf = grads_activation(V, H_attn, H_hopf)
            V = V - step_v * dE_dV
            H_attn = H_attn - step_h * dE_dF_attn
            H_hopf = H_hopf - step_h * dE_dF_hopf
            return (V, H_attn, H_hopf)

        V_T, H_attn_T, H_hopf_T = jax.lax.fori_loop(
            0, shakespeare_config.n_steps, body, (V0, H_attn0, H_hopf0)
        )
        dE_dV_T, _, _ = grads_activation(V_T, H_attn_T, H_hopf_T)
        F_T = -(1.0 / shakespeare_config.tau_v) * dE_dV_T
        return V_T, F_T, H_attn_T, H_hopf_T


    def loss_and_metrics(params, ctx_bits, labels, force_weight):
        B = ctx_bits.shape[0]
        V0 = jnp.zeros((B, shakespeare_config.D), dtype=jnp.float32)
        V_T, F_T, H_attn_T, H_hopf_T = infer_forward_euler_with_force_and_hidden(
            params, V0, ctx_bits
        )
        logits_T = logits_from_v(params, V_T)
        ce = optax.softmax_cross_entropy_with_integer_labels(logits_T, labels).mean()
        force_pen = jnp.mean(jnp.sum(F_T * F_T, axis=1))
        loss = ce + force_weight * force_pen

        attn = jax.nn.softmax(shakespeare_config.beta * H_attn_T, axis=-1)
        attn_entropy = -jnp.sum(attn * jnp.log(jnp.maximum(attn, 1e-20)), axis=-1)
        attn_entropy = jnp.mean(attn_entropy)
        attn_max = jnp.mean(jnp.max(attn, axis=-1))

        hopf_act = jnp.maximum(H_hopf_T, 0.0)
        hopf_sat = jnp.mean((hopf_act > 0.0).astype(jnp.float32))

        v_rms = jnp.sqrt(jnp.mean(jnp.sum(V_T * V_T, axis=1)))
        force_rms = jnp.sqrt(jnp.mean(jnp.sum(F_T * F_T, axis=1)))
        force_rel = force_rms / jnp.maximum(v_rms, 1e-8)

        metrics = dict(
            ce=ce,
            force_pen=force_pen,
            v_rms=v_rms,
            force_rms=force_rms,
            force_rel=force_rel,
            attn_entropy=attn_entropy,
            attn_max=attn_max,
            hopf_sat=hopf_sat,
        )
        return loss, metrics


    @jax.jit
    def train_step(params, opt_state, train_X, train_Y, force_weight):
        (loss, metrics), grads = jax.value_and_grad(loss_and_metrics, has_aux=True)(
            params, train_X, train_Y, force_weight
        )
        grad_norm = optax.global_norm(grads)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        update_norm = optax.global_norm(updates)
        param_norm = optax.global_norm(params)
        relative_step_size = update_norm / jnp.maximum(param_norm, 1e-8)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, grad_norm, relative_step_size, metrics


    losses_all = []
    losses_steps: List[int] = []
    accs_all = []
    accs_steps: List[int] = []
    perplexities_all = []
    perplexities_steps: List[int] = []
    generated_texts_all = []
    generated_texts_epochs: List[int] = []

    global_step = 0
    
    # Prepare a seed context for text generation (use first sequence from validation)
    seed_context = valid_X[0]

    epoch = 0
    done = False

    while not done:
        t_start = time.time()
        key, key_perm = jr.split(key)
        index_perm = jr.permutation(key_perm, num_train)
        train_X_epoch = train_X[index_perm]
        train_y_epoch = train_y[index_perm]
        losses_epoch = []

        lam_force = jnp.asarray(force_penalty_weight(global_step, shakespeare_config), dtype=jnp.float32)

        for batch in range(num_batches):
            if global_step >= shakespeare_config.max_steps:
                done = True
                break

            start = batch * shakespeare_config.batch_size
            stop = min((batch + 1) * shakespeare_config.batch_size, num_train)
            key, sub = jr.split(key)
            batch_train_X = train_X_epoch[start:stop]
            batch_train_y = train_y_epoch[start:stop]

            params, opt_state, loss, grad_norm, relative_step_size, metrics = train_step(
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
            wandb.log(
                {
                    "train/loss": float(loss),
                    "train/ce": float(metrics["ce"]),
                    "train/force_pen": float(metrics["force_pen"]),
                    "train/force_w": float(lam_force),
                    "train/v_rms": float(metrics["v_rms"]),
                    "train/force_rms": float(metrics["force_rms"]),
                    "train/force_rel": float(metrics["force_rel"]),
                    "train/attn_entropy": float(metrics["attn_entropy"]),
                    "train/attn_max": float(metrics["attn_max"]),
                    "train/hopf_sat": float(metrics["hopf_sat"]),
                    "train/grad_norm": float(grad_norm),
                    "train/relative_step_size": float(relative_step_size),
                    "train/lr_fast": lr_fast,
                    "train/lr_slow": lr_slow,
                    "epoch": epoch,
                },
                step=global_step,
            )
            # ------------------------------------

            global_step += 1

        t_end = time.time()
        epoch_time = t_end - t_start

        if not losses_epoch:
            break

        mean_train_loss = float(jnp.mean(jnp.array(losses_epoch)))

        # Log epoch aggregates
        wandb.log(
            {
                "epoch/train_loss_mean": mean_train_loss,
                "epoch/time_sec": float(epoch_time),
                "epoch": epoch,
            },
            step=global_step - 1,
        )

        acc = evaluate(params, valid_X, valid_y, shakespeare_config)
        accs_all.append(float(acc))
        accs_steps.append(global_step - 1)

        # Calculate perplexity
        perplexity = calculate_perplexity(params, valid_X, valid_y, shakespeare_config)
        perplexities_all.append(float(perplexity))
        perplexities_steps.append(global_step - 1)

        # Generate text
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

        # Show seed context for reference
        seed_text = "".join([idx_to_char[int(idx)] for idx in seed_context])

        print(
            f"epoch {epoch:5d} | "
            f"step {global_step:6d}/{shakespeare_config.max_steps} | "
            f"force_w {float(lam_force):5.3f} | "
            f"train loss {mean_train_loss:8.4f} | "
            f"valid acc {float(acc):6.4f} | "
            f"perplexity {perplexity:8.4f} | "
            f"{epoch_time:3.3f}s / epoch"
        )
        print(f"  Seed: '{seed_text}'")
        print(f"  Generated: '{generated_text}'")

        # ---- W&B eval logging ----
        wandb.log(
            {
                "valid/accuracy": float(acc),
                "valid/perplexity": float(perplexity),
                "valid/eval_epoch": epoch,
                "generated_text": wandb.Html(f"<pre>Seed: {seed_text}\nGenerated: {generated_text}</pre>"),
            },
            step=global_step - 1,
        )
        # --------------------------------

        save_params(params, f"{output_dir}/model_shakespeare.npz")
        if acc >= 0.5:
            save_params(params, f"{output_dir}/model_shakespeare_best.npz")

        epoch += 1

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
