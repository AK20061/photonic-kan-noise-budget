# the 5 experiments. success criteria were written down BEFORE each one was run (see README)

import numpy as np
import pandas as pd
import torch

from . import data, sdk
from .models import MLP, make_kan, n_params
from .train import (Split, fit_sensitivity, make_split, noise_curve, predict, rmse,
                    sigma_budget, test_rmse, threshold_crossing, train)


def _sigmas(cfg):
    return np.linspace(0, cfg.sigma_max, cfg.sigma_n)


def _kan(cfg):
    return make_kan(cfg.hidden, cfg.k1, cfg.k2)


# ---------------- 1. capacity ----------------
# unregularised KANs vs additive KAN vs MLPs, accuracy vs size. one split

def capacity(X, y, cfg):
    sp = make_split(y, 0, cfg.train_frac)
    models = [("KAN-additive", lambda K=K: make_kan(0, K)) for K in (3, 6)]
    models += [("KAN-unregularised", lambda H=H, K=K: make_kan(H, K))
               for H, K in [(2, 3), (4, 3), (4, 5), (8, 5), (8, 8), (16, 8)]]
    models += [("MLP", lambda H=H: MLP(H)) for H in (4, 8, 16, 32, 64)]
    # (the H=H, K=K defaults matter - without them every lambda uses the last loop value)

    rows = []
    for name, make in models:
        for seed in range(cfg.n_seeds):
            torch.manual_seed(seed)
            m = train(make(), X, y, sp.fit, sp.mu, sp.sd, steps=cfg.steps, lr=cfg.lr)
            pred = predict(m, X, sp.mu, sp.sd)
            rows.append(dict(model=name, params=n_params(m),
                             train_rmse=rmse(pred, y, sp.fit), test_rmse=rmse(pred, y, sp.test)))
    return pd.DataFrame(rows)


# ---------------- 2. lambda selection ----------------
# cap k2, add smoothness penalty. lambda picked on a validation chunk of the TRAIN set,
# test set used once at the end

def lambda_selection(X, y, cfg):
    sp = make_split(y, 0, cfg.train_frac)
    n_val = round(0.2 * len(sp.fit))
    val, fit2 = sp.fit[:n_val], sp.fit[n_val:]
    val_sp = Split(fit2, val, sp.mu, sp.sd)

    rows = []
    for H in (2, 4, 8):
        for lam in (1e-4, 1e-3, 1e-2):
            for seed in range(cfg.n_seeds):
                torch.manual_seed(seed)
                m = train(make_kan(H, cfg.k1, cfg.k2), X, y, fit2, sp.mu, sp.sd,
                          lam=lam, steps=cfg.steps, lr=cfg.lr)
                rows.append(dict(hidden=H, lam=lam, params=n_params(m),
                                 val_rmse=test_rmse(m, X, y, val_sp)))
    val_df = pd.DataFrame(rows)

    means = val_df.groupby(["hidden", "lam", "params"], as_index=False).val_rmse.mean()
    best = means.loc[means.groupby("hidden").val_rmse.idxmin()]

    rows = []
    for _, b in best.iterrows():
        for seed in range(cfg.n_seeds):
            torch.manual_seed(seed)
            m = train(make_kan(int(b.hidden), cfg.k1, cfg.k2), X, y, sp.fit, sp.mu, sp.sd,
                      lam=b.lam, steps=cfg.steps, lr=cfg.lr)
            rows.append(dict(hidden=int(b.hidden), lam=b.lam, params=n_params(m),
                             test_rmse=test_rmse(m, X, y, sp)))
    return val_df, pd.DataFrame(rows)


# ---------------- 3. headline ----------------
# locked KAN vs MLP over splits x seeds: fp32 vs bf16, SDK parity, noise budget

def headline(X, y, cfg, use_sdk=True):
    sig = _sigmas(cfg)
    rows, curves = [], []
    for s in range(cfg.n_splits):
        sp = make_split(y, s, cfg.train_frac)
        for seed in range(cfg.n_seeds):
            torch.manual_seed(seed)
            kan = train(_kan(cfg), X, y, sp.fit, sp.mu, sp.sd, lam=cfg.lam, steps=cfg.steps, lr=cfg.lr)
            torch.manual_seed(seed)
            mlp = train(MLP(cfg.mlp_hidden), X, y, sp.fit, sp.mu, sp.sd, steps=cfg.steps, lr=cfg.lr)

            r = dict(split=s, seed=seed)
            r["kan_fp32"] = test_rmse(kan, X, y, sp)
            r["kan_bf16"] = test_rmse(kan, X, y, sp, dtype=torch.bfloat16)
            r["mlp_fp32"] = test_rmse(mlp, X, y, sp)
            r["mlp_bf16"] = test_rmse(mlp, X, y, sp, dtype=torch.bfloat16)
            r["kan_bf16_cost"] = r["kan_bf16"] - r["kan_fp32"]
            r["mlp_bf16_cost"] = r["mlp_bf16"] - r["mlp_fp32"]
            if use_sdk and sdk.AVAILABLE:
                r["kan_sdk"] = rmse(sdk.sdk_predict(kan, X, sp.mu, sp.sd), y, sp.test)

            curve = noise_curve(kan, X, y, sp, sig, cfg.noise_reps)
            e0, S = fit_sensitivity(sig, curve)
            r["sensitivity"] = S
            r["sigma_10pct"] = sigma_budget(e0, S, 1.1 * e0)
            r["sigma_1kcal"] = sigma_budget(e0, S, 1.0)
            rows.append(r)
            curves.append(curve)
    return pd.DataFrame(rows), np.array(curves)


# ---------------- 4. noise-aware training ----------------
# train with jitter switched on. target was 2x more tolerated noise -> it FAILED (1.26x),
# which is what led to the floor argument in exp 5

def noise_aware(X, y, cfg, train_sigmas=(0.0, 0.03, 0.06, 0.1)):
    sig = np.linspace(0, 0.3, 16)
    low = sig <= 0.1
    rows, curves = [], {}
    for s in range(cfg.n_splits):
        sp = make_split(y, s, cfg.train_frac)
        for seed in range(min(cfg.n_seeds, 2)):
            for st in train_sigmas:
                torch.manual_seed(seed)
                kan = train(_kan(cfg), X, y, sp.fit, sp.mu, sp.sd, lam=cfg.lam,
                            steps=cfg.steps, lr=cfg.lr, train_sigma=st)
                curve = noise_curve(kan, X, y, sp, sig, cfg.noise_reps)
                _, S = fit_sensitivity(sig[low], curve[low])
                curves.setdefault(st, []).append(curve)
                rows.append(dict(split=s, seed=seed, train_sigma=st, clean=curve[0], sensitivity=S,
                                 sigma_at_0p5=threshold_crossing(sig, curve, 0.5),
                                 sigma_at_1=threshold_crossing(sig, curve, 1.0)))
    curves = {k: (sig, np.array(v)) for k, v in curves.items()}
    return pd.DataFrame(rows), curves


# ---------------- 5. sensitivity vs spread ----------------
# same locked model on 6 landscapes. prediction: S / spread ~ constant (+-15%) and >= 1

def sensitivity_vs_spread(landscapes, cfg):
    sig = _sigmas(cfg)
    rows = []
    for name, df in landscapes.items():
        X, y = data.arrays(df)
        for s in range(cfg.n_splits):
            sp = make_split(y, s, cfg.train_frac)
            for seed in range(min(cfg.n_seeds, 2)):
                torch.manual_seed(seed)
                kan = train(_kan(cfg), X, y, sp.fit, sp.mu, sp.sd, lam=cfg.lam, steps=cfg.steps, lr=cfg.lr)
                curve = noise_curve(kan, X, y, sp, sig, cfg.noise_reps)
                _, S = fit_sensitivity(sig, curve)
                rows.append(dict(landscape=name, clean=curve[0], sensitivity=S))
    runs = pd.DataFrame(rows)

    M = runs.groupby("landscape").agg(clean=("clean", "mean"),
                                      sensitivity=("sensitivity", "mean"),
                                      sensitivity_sd=("sensitivity", "std"))
    M.insert(0, "spread", [data.spread(data.arrays(landscapes[n])[1]) for n in M.index])
    M.insert(1, "fourier_rmse", [data.fourier_baseline_rmse(landscapes[n]) for n in M.index])
    M["kappa"] = M.sensitivity / M.spread
    return runs, M.sort_values("spread")


def budget_table(kappa, landscapes, targets=(0.5, 1.0), clean=0.45):
    # sigma_max = sqrt(T^2 - E0^2) / (kappa * s)
    rows = []
    for name, df in landscapes.items():
        s = data.spread(data.arrays(df)[1])
        r = dict(landscape=name, spread=s)
        for t in targets:
            r[f"sigma_max_T={t}"] = sigma_budget(clean, kappa * s, t)
        rows.append(r)
    return pd.DataFrame(rows).sort_values("spread")
