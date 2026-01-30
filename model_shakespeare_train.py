import os
from typing import List

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import time

import jax
import jax.numpy as jnp
import jax.random as jr
import optax

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
    
    # Create a new config instance for Shakespeare training
    vocab_size = len(char_to_idx)
    print(f"Using vocab_size={vocab_size}")
    
    # Create Shakespeare-specific config
    shakespeare_config = Config(
        L=16,  # context length for Shakespeare
        vocab_size=vocab_size,
    )
    
    key = jr.PRNGKey(shakespeare_config.seed)
    
    # Initialize parameters with correct vocab_size
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

    # don't apply weight decay to xi_attn_embed_raw and xi_hopf_raw
    tx_fast = optax.chain(
        optax.clip_by_global_norm(shakespeare_config.max_norm),
        optax.adamw(
            learning_rate=lr_sched(shakespeare_config.lr_peak_value),
            weight_decay=shakespeare_config.fast_weight_decay,
        ),
    )
    tx_slow = optax.chain(
        optax.clip_by_global_norm(shakespeare_config.max_norm),
        optax.adamw(
            learning_rate=lr_sched(shakespeare_config.lr_peak_value),
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
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        return params, opt_state, loss

    losses_all = []
    losses_steps: List[int] = []
    accs_all = []
    accs_steps: List[int] = []
    for epoch in range(shakespeare_config.train_epochs):
        t_start = time.time()
        key, key_perm = jr.split(key)
        index_perm = jr.permutation(key_perm, num_train)
        train_X_epoch = train_X[index_perm]
        train_y_epoch = train_y[index_perm]
        losses_epoch = []

        # compute current force penalty weight (cosine ramp 0 -> 1)
        lam_force = jnp.asarray(force_penalty_weight(epoch, shakespeare_config), dtype=jnp.float32)

        for batch in range(num_batches):
            start = batch * shakespeare_config.batch_size
            stop = min((batch + 1) * shakespeare_config.batch_size, num_train)
            key, sub = jr.split(key)
            batch_train_X = train_X_epoch[start:stop]
            batch_train_y = train_y_epoch[start:stop]

            params, opt_state, loss = train_step(
                params,
                opt_state,
                batch_train_X,
                batch_train_y,
                force_weight=lam_force,
            )
            losses_epoch.append(loss)
            losses_all.append(float(loss))
            losses_steps.append(epoch * num_batches + batch)

        t_end = time.time()

        if epoch % 100 == 0:
            acc = evaluate(params, valid_X, valid_y, shakespeare_config)
            accs_all.append(float(acc))
            accs_steps.append((epoch + 1) * num_batches - 1)
            print(
                f"epoch {epoch:5d} | "
                f"force_w {float(lam_force):5.3f} | "
                f"train loss {jnp.mean(jnp.array(losses_epoch)):8.4f} | "
                f"valid acc {acc:6.4f} | "
                f"{(t_end - t_start):3.3f}s / epoch | "
            )
            save_params(params, "data/model_shakespeare.npz")

            if acc >= 0.99:
                save_params(params, "data/model_shakespeare_best.npz")

    save_metrics({"step": losses_steps, "loss": losses_all}, "data/losses_shakespeare.json")
    save_metrics({"step": accs_steps, "accuracy": accs_all}, "data/accs_shakespeare.json")

    save_params(params, "data/model_shakespeare.npz")
    print("Training complete. Model parameters saved to 'data/model_shakespeare.npz'.")
