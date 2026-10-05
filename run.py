# runs everything in the README
#   python run.py                  all 5 experiments (~30-45 min on cpu)
#   python run.py --exp headline   just one
#   python run.py --quick          smoke test (~3 min once scans are cached), numbers meaningless

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from pkan import data, experiments, plots, sdk
from pkan.config import DEFAULT, QUICK

EXPS = ["capacity", "lambda", "headline", "noise_aware", "spread"]
pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 30)


def header(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def mean_sd(df, by, cols):
    return df.groupby(by)[cols].agg(["mean", "std"]).round(3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=EXPS + ["all"], default="all")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-sdk", action="store_true")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    cfg = QUICK if args.quick else DEFAULT
    out, cache = Path(args.out), Path(args.data)
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    todo = EXPS if args.exp == "all" else [args.exp]
    t0 = time.time()

    print("config:", cfg)
    print("Q.ANT SDK available:", sdk.AVAILABLE)
    ala = data.load_molssi(cache)
    X, y = data.arrays(ala)
    print(f"alanine dipeptide (MolSSI): {len(y)} pts, E in [{y.min():.2f}, {y.max():.2f}] kcal/mol, "
          f"spread {data.spread(y):.2f}")

    # ---- 1 ----
    if "capacity" in todo:
        header("1. CAPACITY  (test RMSE kcal/mol vs params, 1 split, mean/std over seeds)")
        df = experiments.capacity(X, y, cfg)
        df.to_csv(out / "capacity.csv", index=False)
        print(mean_sd(df, ["model", "params"], ["train_rmse", "test_rmse"]).to_string())
        print("\nunregularised 2-layer KANs memorise (train -> 0, test gets worse with size).")
        print("additive KAN plateaus ~4 - f(phi)+g(psi) can't do phi-psi coupling.")
        plots.capacity(df, figs / "capacity.png")

    # ---- 2 ----
    if "lambda" in todo:
        header("2. REGULARISED KAN  (lambda picked on validation, test used once)")
        val_df, test_df = experiments.lambda_selection(X, y, cfg)
        val_df.to_csv(out / "lambda_validation.csv", index=False)
        test_df.to_csv(out / "lambda_test.csv", index=False)
        print(test_df.groupby(["hidden", "lam", "params"]).test_rmse.agg(["mean", "std"]).round(3).to_string())

    # ---- 3 ----
    if "headline" in todo:
        header(f"3. HEADLINE  KAN(H={cfg.hidden}, K={cfg.k1}/{cfg.k2}, lam={cfg.lam}) vs MLP({cfg.mlp_hidden}), "
               f"{cfg.n_splits} splits x {cfg.n_seeds} seeds")
        R, curves = experiments.headline(X, y, cfg, use_sdk=not args.no_sdk)
        R.to_csv(out / "headline_runs.csv", index=False)
        print(R.drop(columns=["split", "seed"]).agg(["mean", "std"]).T.round(3).to_string())
        if "kan_sdk" in R:
            gap = (R.kan_sdk - R.kan_bf16).abs()
            print(f"\nSDK vs torch bf16: mean |diff| {gap.mean():.4f}, max {gap.max():.4f} kcal/mol")
        print(f"\nnoise budget on this landscape:")
        print(f"  S = {R.sensitivity.mean():.2f} +- {R.sensitivity.std():.2f} kcal/mol/rad")
        print(f"  sigma for <=10% degradation: {R.sigma_10pct.mean():.3f} +- {R.sigma_10pct.std():.3f} rad")
        print(f"  sigma for <=1 kcal/mol:      {R.sigma_1kcal.mean():.3f} +- {R.sigma_1kcal.std():.3f} rad")
        plots.noise_tolerance(np.linspace(0, cfg.sigma_max, cfg.sigma_n), curves, figs / "noise_tolerance.png")

    # ---- 4 ----
    if "noise_aware" in todo:
        header("4. NOISE-AWARE TRAINING")
        N, curves = experiments.noise_aware(X, y, cfg)
        N.to_csv(out / "noise_aware.csv", index=False)
        print(mean_sd(N, "train_sigma", ["clean", "sensitivity", "sigma_at_0p5", "sigma_at_1"]).to_string())
        S0 = N[N.train_sigma == 0].sensitivity.mean()
        Smin = N.groupby("train_sigma").sensitivity.mean().min()
        floor = data.spread(y)
        print(f"\nS: {S0:.2f} -> {Smin:.2f} kcal/mol/rad, floor ~ spread = {floor:.2f}"
              f"  ({(S0 - Smin) / (S0 - floor):.0%} of the gap closed)")
        plots.noise_aware(curves, figs / "noise_aware.png")

    # ---- 5 ----
    if "spread" in todo:
        header("5. SENSITIVITY vs SPREAD  (6 landscapes)")
        land = {n: data.load_landscape(n, cache) for n in data.ALL_LANDSCAPES}
        runs, M = experiments.sensitivity_vs_spread(land, cfg)
        runs.to_csv(out / "spread_runs.csv", index=False)
        M.to_csv(out / "sensitivity_vs_spread.csv")
        print(M.round(3).to_string())
        k = M.kappa.mean()
        cv = M.kappa.std() / k
        ok15 = M.kappa.between(0.85 * k, 1.15 * k).all()
        print(f"\nkappa = S/spread = {k:.2f}, CV {cv:.1%}, all within +-15%: {ok15}, all >= 1: {(M.kappa >= 1).all()}")
        print("(fourier_rmse = data sanity check, should be well under 1)")

        B = experiments.budget_table(k, land)
        B.to_csv(out / "noise_budget.csv", index=False)
        print(f"\nBUDGET  sigma_max = sqrt(T^2 - E0^2) / (kappa*s),  kappa={k:.2f}, E0=0.45")
        print(B.round(3).to_string(index=False))
        plots.sensitivity_vs_spread(M, figs / "sensitivity_vs_spread.png")

    print(f"\ndone in {(time.time() - t0) / 60:.1f} min -> {out.resolve()}")


if __name__ == "__main__":
    main()
