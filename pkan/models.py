# photonic KAN + MLP baseline
#
# KANLayer computes exactly what Q.ANT's calc_kan_layer_fprop does:
#     y_j = sum_{i,l} tcos(k_l * x_i + phi_jil) * amp_jil   (+ bias)
# so trained weights can be pushed straight through their SDK.
# tcos is just cos here - checked the SDK cpu backend, max diff ~0.002 (bf16 rounding)

import contextlib
import math

import torch
import torch.nn as nn

_SIGMA = 0.0   # phase noise (rad), set via phase_noise()


@contextlib.contextmanager
def phase_noise(sigma):
    # iid gaussian jitter on every tcos argument inside the with-block
    global _SIGMA
    old, _SIGMA = _SIGMA, float(sigma)
    try:
        yield
    finally:
        _SIGMA = old


def tcos(x):
    if _SIGMA > 0:
        x = x + _SIGMA * torch.randn_like(x)
    return torch.cos(x)


class KANLayer(nn.Module):
    def __init__(self, n_in, n_out, n_freq):
        super().__init__()
        self.register_buffer("ks", torch.arange(1, n_freq + 1, dtype=torch.float32))
        self.phi = nn.Parameter(2 * math.pi * torch.rand(n_out, n_in, n_freq))
        self.amp = nn.Parameter(torch.randn(n_out, n_in, n_freq) / math.sqrt(n_in * n_freq))
        self.b = nn.Parameter(torch.zeros(n_out))

    def forward(self, x):
        # x: (B, in) -> arg: (B, out, in, K)
        arg = x[:, None, :, None] * self.ks + self.phi
        return (tcos(arg) * self.amp).sum((2, 3)) + self.b


def make_kan(hidden, k1, k2=None):
    # hidden=0 -> single layer = f(phi) + g(psi), can't do coupling (used as a control)
    if hidden == 0:
        return nn.Sequential(KANLayer(2, 1, k1))
    return nn.Sequential(KANLayer(2, hidden, k1), KANLayer(hidden, 1, k2 or k1))


class MLP(nn.Module):
    # feed it cos/sin so it also knows the angles wrap - otherwise the comparison is rigged
    def __init__(self, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(torch.cat([torch.cos(x), torch.sin(x)], 1))


def n_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def smoothness_penalty(model):
    # sum (k*amp)^2 -> punishes high freq content. MLP gets 0
    terms = [((m.amp * m.ks) ** 2).sum() for m in model.modules() if isinstance(m, KANLayer)]
    if not terms:
        return torch.zeros(())
    return torch.stack(terms).sum()
