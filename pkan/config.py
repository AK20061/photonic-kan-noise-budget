from dataclasses import dataclass, replace


# hyperparams locked after the lambda sweep (exp 2) - don't touch these when rerunning
@dataclass(frozen=True)
class Config:
    hidden: int = 8
    k1: int = 5          # freqs in layer 1 (integer k -> exact 2pi periodicity in the angles)
    k2: int = 2          # freqs in layer 2. keep LOW - stacked cosines multiply frequency
    lam: float = 1e-3    # smoothness penalty
    mlp_hidden: int = 8  # = 121 params
    steps: int = 4000
    lr: float = 1e-2
    train_frac: float = 0.8
    n_splits: int = 3
    n_seeds: int = 3
    noise_reps: int = 10
    sigma_max: float = 0.1
    sigma_n: int = 11


DEFAULT = Config()
QUICK = replace(DEFAULT, steps=600, n_splits=1, n_seeds=1, noise_reps=3)   # smoke test, numbers are junk
