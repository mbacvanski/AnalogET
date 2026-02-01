import argparse
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
        help="Use only 100 batches of data for quick hyperparameter search",
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
    
    # Apply sample mode if requested (limit to 100 batches worth of data)
    if args.sample_mode:
        max_samples = 100 * 256  # 100 batches * batch_size
        if len(train_X) > max_samples:
            print(f"Sample mode: Using {max_samples} training samples (100 batches)")
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


    # ---- create schedule objects so we can log LR ----
    lr_schedule_fast = lr_sched(shakespeare_config.lr_peak_value)
    lr_schedule_slow = lr_sched(shakespeare_config.lr_peak_value)
    # -------------------------------------------------------

    tx_fast = optax.chain(
        optax.clip_by_global_norm(shakespeare_config.max_norm),
        optax.adamw(
            learning_rate=lr_schedule_fast,  # <-- CHANGE: use object
            weight_decay=shakespeare_config.fast_weight_decay,
        ),
    )
    tx_slow = optax.chain(
        optax.clip_by_global_norm(shakespeare_config.max_norm),
        optax.adamw(
            learning_rate=lr_schedule_slow,  # <-- CHANGE: use object
            weight_decay=shakespeare_config.slow_weight_decay,
        ),
    )

    optimizer = optax.multi_transform(
        {"fast": tx_fast, "slow": tx_slow}, label_tree(params)
    )
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
    
    # Prepare a seed context for text generation (use first sequence from validation)
    seed_context = valid_X[0]

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
            wandb.log(
                {
                    "train/loss": float(loss),
                    "train/force_w": float(lam_force),
                    "train/grad_norm": float(grad_norm),
                    "train/lr_fast": float(lr_schedule_fast(global_step)),
                    "train/lr_slow": float(lr_schedule_slow(global_step)),
                    "epoch": epoch,
                },
                step=global_step,
            )
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

            save_params(params, "data/shakespeare/model_shakespeare.npz")
            if acc >= 0.5:
                save_params(params, "data/shakespeare/model_shakespeare_best.npz")

    save_metrics({"step": losses_steps, "loss": losses_all}, "data/shakespeare/losses_shakespeare.json")
    save_metrics({"step": accs_steps, "accuracy": accs_all}, "data/shakespeare/accs_shakespeare.json")
    save_metrics({"step": perplexities_steps, "perplexity": perplexities_all}, "data/shakespeare/perplexities_shakespeare.json")
    save_metrics({"epoch": generated_texts_epochs, "text": generated_texts_all}, "data/shakespeare/generated_texts_shakespeare.json")

    save_params(params, "data/shakespeare/model_shakespeare.npz")
    print("Training complete. Model parameters saved to 'data/shakespeare/model_shakespeare.npz'.")
    
    # Create plots
    print("Creating training plots...")
    plot_path = plot_training_metrics(
        losses_steps, losses_all,
        accs_steps, accs_all,
        perplexities_steps, perplexities_all,
        output_path="data/shakespeare/training_metrics_shakespeare.png"
    )
    print(f"Plots saved to {plot_path}")
    
    # Log final plot to W&B
    wandb.log({"training_summary": wandb.Image(plot_path)})

    wandb.finish()
