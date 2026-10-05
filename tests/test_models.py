import math

import numpy as np
import pytest
import torch

from pkan.models import MLP, make_kan, n_params, phase_noise, smoothness_penalty


def test_kan_is_periodic_in_both_angles():
    torch.manual_seed(0)
    m = make_kan(8, 5, 2)
    x = torch.rand(16, 2) * 6 - 3
    for d in range(2):
        shift = torch.zeros(2)
        shift[d] = 2 * math.pi
        assert torch.allclose(m(x), m(x + shift), atol=1e-4)


def test_param_counts_match_readme():
    assert n_params(make_kan(8, 5, 2)) == 201
    assert n_params(MLP(8)) == 121


def test_noise_context_restores_state():
    torch.manual_seed(0)
    m, x = make_kan(4, 3), torch.rand(8, 2)
    clean = m(x)
    with phase_noise(0.1):
        assert not torch.allclose(m(x), clean)
    assert torch.allclose(m(x), clean)


def test_penalty_only_on_kan():
    assert smoothness_penalty(MLP(4)).item() == 0
    assert smoothness_penalty(make_kan(4, 3)).item() > 0


def test_sdk_parity():
    sdk = pytest.importorskip("pkan.sdk")
    if not sdk.AVAILABLE:
        pytest.skip("Q.ANT SDK not installed")
    torch.manual_seed(0)
    m = make_kan(8, 5, 2)
    X = np.random.default_rng(0).uniform(-math.pi, math.pi, (64, 2)).astype(np.float32)
    with torch.no_grad():
        ref = m(torch.tensor(X)).squeeze(-1).numpy()
    out = sdk.sdk_predict(m, X, 0.0, 1.0)
    assert np.abs(out - ref).max() < 0.1   # bfloat16 rounding on the SDK side
