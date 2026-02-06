import argparse
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
from model import evaluate, force_penalty_weight, init_params, label_tree, loss_fn
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
        "--ctx_length",
        type=int,
        default=16,
        help="Context length for training sequences (default: 16)",
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
    args = parser.parse_args()
    
    # Load or prepare Shakespeare dataset
    filename_prefix = "shakespeare_data"
    ctx_length = args.ctx_length
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

    # Initialize config from JSON file if provided, otherwise use defaults
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
    else:
        shakespeare_config = Config(
            L=ctx_length,
            vocab_size=vocab_size,
        )

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
    total_steps = shakespeare_config.train_epochs * num_batches


    def lr_sched(
            peak: float, warmup_steps: int = 0, end_factor: float = 0.3
    ) -> optax.Schedule:
        """warm up to peak, then decay to peak * end_factor"""
        return optax.warmup_cosine_decay_schedule(
            init_value=shakespeare_config.lr_init_value,
            peak_value=peak,
            warmup_steps=warmup_steps,
            decay_steps=max(1, total_steps - warmup_steps),
            end_value=peak * end_factor,
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


    @jax.jit
    def train_step(params, opt_state, train_X, train_Y, force_weight):
        loss, grads = jax.value_and_grad(loss_fn)(
            params, train_X, train_Y, force_weight, shakespeare_config
        )
        grad_norm = optax.global_norm(grads)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, grad_norm


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
        t_start = time.time()
        key, key_perm = jr.split(key)
        index_perm = jr.permutation(key_perm, num_train)
        train_X_epoch = train_X[index_perm]
        train_y_epoch = train_y[index_perm]
        losses_epoch = []

        lam_force = jnp.asarray(force_penalty_weight(epoch, shakespeare_config), dtype=jnp.float32)

        for batch in range(num_batches):
            start = batch * shakespeare_config.batch_size
            stop = min((batch + 1) * shakespeare_config.batch_size, num_train)
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

        if epoch % 1 == 0:
            acc = evaluate(params, valid_X, valid_y, shakespeare_config)
            accs_all.append(float(acc))
            accs_steps.append((epoch + 1) * num_batches - 1)
            
            # Calculate perplexity
            perplexity = calculate_perplexity(params, valid_X, valid_y, shakespeare_config)
            perplexities_all.append(float(perplexity))
            perplexities_steps.append((epoch + 1) * num_batches - 1)
            
            # Generate text from training example
            key, gen_key_train = jr.split(key)
            generated_text_train = generate_text(
                params,
                seed_context_train,
                idx_to_char,
                shakespeare_config,
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
                shakespeare_config,
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

            print(
                f"epoch {epoch:5d} | "
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
            wandb.log(
                {
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
                },
                step=global_step - 1,
            )
            # --------------------------------

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
