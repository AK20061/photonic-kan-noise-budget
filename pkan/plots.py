import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _save(fig, path):
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def capacity(df, path):
    g = df.groupby(["model", "params"]).test_rmse.mean().reset_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, d in g.groupby("model"):
        ax.plot(d.params, d.test_rmse, "o-", label=name)
    ax.set_xscale("log")
    ax.set_xlabel("parameters")
    ax.set_ylabel("test RMSE (kcal/mol)")
    ax.set_title("accuracy vs size (unregularised KANs)")
    ax.legend()
    _save(fig, path)


def noise_tolerance(sigmas, curves, path):
    m, s = curves.mean(0), curves.std(0)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(sigmas, m, "o-", label=f"photonic KAN ({len(curves)} runs)")
    ax.fill_between(sigmas, m - s, m + s, alpha=0.3)
    ax.axhline(1.0, ls="--", c="grey", label="1 kcal/mol")
    ax.set_xlabel("phase noise σ (rad)")
    ax.set_ylabel("test RMSE (kcal/mol)")
    ax.legend()
    _save(fig, path)


def noise_aware(curves, path):
    fig, ax = plt.subplots(figsize=(6, 4))
    for st, (sig, c) in curves.items():
        ax.plot(sig, c.mean(0), "o-", ms=3, label=f"trained at σ={st}")
    ax.axhline(1.0, ls="--", c="grey")
    ax.axhline(0.5, ls=":", c="grey")
    ax.set_xlabel("phase noise at inference σ (rad)")
    ax.set_ylabel("test RMSE (kcal/mol)")
    ax.legend()
    _save(fig, path)


def sensitivity_vs_spread(M, path):
    k = M.kappa.mean()
    xs = np.linspace(0, M.spread.max() * 1.1, 50)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.errorbar(M.spread, M.sensitivity, yerr=M.sensitivity_sd, fmt="o", capsize=4)
    for name, r in M.iterrows():
        ax.annotate(name, (r.spread, r.sensitivity), textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.plot(xs, k * xs, "--", label=f"S = {k:.2f} × spread")
    ax.plot(xs, xs, ":", c="grey", label="first-order floor (κ = 1)")
    ax.set_xlabel("landscape spread s = std(E) (kcal/mol)")
    ax.set_ylabel("phase-noise sensitivity S (kcal/mol/rad)")
    ax.legend()
    _save(fig, path)
