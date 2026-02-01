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
from utils import save_metrics, save_params

if __name__ == "__main__":
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

    global_step = 0

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

            print(
                f"epoch {epoch:5d} | "
                f"force_w {float(lam_force):5.3f} | "
                f"train loss {mean_train_loss:8.4f} | "
                f"valid acc {float(acc):6.4f} | "
                f"{epoch_time:3.3f}s / epoch | "
            )

            # ---- W&B eval logging ----
            wandb.log(
                {
                    "valid/accuracy": float(acc),
                    "valid/eval_epoch": epoch,
                },
                step=global_step - 1,
            )
            # --------------------------------

            save_params(params, "data/model_shakespeare.npz")
            if acc >= 0.99:
                save_params(params, "data/model_shakespeare_best.npz")

    save_metrics({"step": losses_steps, "loss": losses_all}, "data/losses_shakespeare.json")
    save_metrics({"step": accs_steps, "accuracy": accs_all}, "data/accs_shakespeare.json")

    save_params(params, "data/model_shakespeare.npz")
    print("Training complete. Model parameters saved to 'data/model_shakespeare.npz'.")

    wandb.finish()
