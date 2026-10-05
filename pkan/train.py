import copy
from dataclasses import dataclass

import numpy as np
import torch
from scipy.optimize import curve_fit

from .models import phase_noise, smoothness_penalty


@dataclass(frozen=True)
class Split:
    fit: np.ndarray
    test: np.ndarray     # only touched for final numbers
    mu: float            # standardisation from train pts only
    sd: float


def make_split(y, seed, train_frac):
    idx = np.random.default_rng(seed).permutation(len(y))
    k = int(train_frac * len(y))
    fit, test = idx[:k], idx[k:]
    return Split(fit, test, float(y[fit].mean()), float(y[fit].std(ddof=1)))


# ---------------- training ----------------

def train(model, X, y, idx, mu, sd, lam=0.0, steps=4000, lr=1e-2, train_sigma=0.0):
    # full batch adam + cosine lr. train_sigma>0 = noise-aware training
    Xt = torch.as_tensor(X[idx])
    target = torch.as_tensor((y[idx] - mu) / sd, dtype=torch.float32)[:, None]
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    with phase_noise(train_sigma):
        for _ in range(steps):
            opt.zero_grad()
            loss = ((model(Xt) - target) ** 2).mean()
            if lam:
                loss = loss + lam * smoothness_penalty(model)
            loss.backward()
            opt.step()
            sched.step()
    return model


# ---------------- eval ----------------

def predict(model, X, mu, sd, dtype=torch.float32, sigma=0.0):
    m = model if dtype == torch.float32 else copy.deepcopy(model).to(dtype)
    with torch.no_grad(), phase_noise(sigma):
        out = m(torch.as_tensor(X).to(dtype)).squeeze(-1).float().numpy()
    return out * sd + mu


def rmse(pred, y, idx):
    return float(np.sqrt(np.mean((pred[idx] - y[idx]) ** 2)))


def test_rmse(model, X, y, split, dtype=torch.float32, sigma=0.0, reps=1):
    if sigma == 0:
        reps = 1   # deterministic, no point repeating
    errs = [rmse(predict(model, X, split.mu, split.sd, dtype=dtype, sigma=sigma), y, split.test)
            for _ in range(reps)]
    return float(np.mean(errs))


def noise_curve(model, X, y, split, sigmas, reps):
    return np.array([test_rmse(model, X, y, split, sigma=s, reps=reps) for s in sigmas])


# ---------------- noise budget ----------------
# error adds in quadrature:  RMSE(sigma) = sqrt(E0^2 + (S*sigma)^2),  S in kcal/mol/rad

def _quad(s, e0, S):
    return np.sqrt(e0 ** 2 + (S * s) ** 2)


def fit_sensitivity(sigmas, curve):
    (e0, S), _ = curve_fit(_quad, np.asarray(sigmas), np.asarray(curve),
                           p0=[float(curve[0]), 9.0], bounds=([0, 0], [np.inf, np.inf]))
    return float(e0), float(S)


def sigma_budget(e0, S, target):
    # biggest sigma that keeps RMSE <= target
    return float(np.sqrt(max(target ** 2 - e0 ** 2, 0.0)) / S)


def threshold_crossing(sigmas, curve, thr):
    # where the measured curve first goes above thr (linear interp). nan if it never does
    curve = np.asarray(curve)
    if curve[0] > thr:
        return 0.0
    above = np.where(curve > thr)[0]
    if len(above) == 0:
        return float("nan")
    i = above[0]
    return float(np.interp(thr, curve[i - 1:i + 1], np.asarray(sigmas)[i - 1:i + 1]))
