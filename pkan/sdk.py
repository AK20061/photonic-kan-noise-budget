# run a trained KAN through Q.ANT's toolkit (optional - everything else works without it)
# cpu-backend wheel: github.com/Q-ANT-GmbH/qant_native_computing_toolkit/releases  (py3.12 only)

import numpy as np

try:
    import ml_dtypes
    from qant_native_computing_toolkit.ai import add_bias_fprop, calc_kan_layer_fprop
    AVAILABLE = True
except ImportError:
    AVAILABLE = False


def _bf16(a):
    if hasattr(a, "detach"):   # torch tensor
        a = a.detach().float().cpu().numpy()
    # NB the SDK throws "Data must be C-contiguous" on column slices from pandas - force it
    return np.ascontiguousarray(np.asarray(a).astype(ml_dtypes.bfloat16))


def sdk_predict(model, X, mu, sd):
    if not AVAILABLE:
        raise RuntimeError("Q.ANT SDK / ml_dtypes not installed")
    h = _bf16(X)
    for layer in model:
        h = np.ascontiguousarray(calc_kan_layer_fprop(h, _bf16(layer.phi), _bf16(layer.amp), _bf16(layer.ks)))
        h = np.ascontiguousarray(add_bias_fprop(h, _bf16(layer.b)))
    return h.astype(np.float32).squeeze(-1) * sd + mu
