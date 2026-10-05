import numpy as np
import pandas as pd

from pkan.data import fourier_baseline_rmse, spread


def test_fourier_baseline_recovers_smooth_surface():
    g = np.arange(-180, 180, 15)
    P, S = np.meshgrid(g, g, indexing="ij")
    E = 3 * np.cos(np.deg2rad(P)) + 2 * np.cos(np.deg2rad(2 * S)) + np.cos(np.deg2rad(P + S))
    df = pd.DataFrame(dict(phi=P.ravel(), psi=S.ravel(), E_kcal=(E - E.min()).ravel()))
    assert fourier_baseline_rmse(df, order=4) < 0.05


def test_spread_is_population_std():
    assert np.isclose(spread(np.array([0.0, 2.0])), 1.0)
