from dataclasses import dataclass, field


@dataclass(frozen=True)
class Config:
    D: int = 16
    L: int = 8
    M: int = 16
    beta: float = 0.1

    xi_attn_embed_raw_scale: float = 0.1
    xi_hopf_raw_scale: float = 0.06

    # --- Stable integration parameters ---
    step_size: float = 1e-3
    T_final: float = 1.0
    n_steps: int = field(init=False)

    batch_size: int = 256
    train_epochs: int = 30_000
    seed: int = 0
    vocab_size: int = 2

    tau_v: float = 1e-1
    tau_h: float = 1e-2

    lr_init_value: float = 0.0
    lr_peak_value: float = 3e-3
    max_norm: float = 1.0
    slow_weight_decay: float = 5e-5
    fast_weight_decay: float = 0.0

    force_penalty_start: int = 20_000
    force_penalty_duration: int = 10_000
    force_penalty_scale: float = 0.05

    def __post_init__(self):
        """Compute n_steps from T_final and step_size."""
        # For frozen dataclasses, we need to use object.__setattr__ to set fields
        object.__setattr__(self, "n_steps", int(self.T_final / self.step_size))


# Create a default instance
config = Config()
