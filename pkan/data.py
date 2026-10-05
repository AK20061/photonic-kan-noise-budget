# energy landscapes E(phi, psi)
#   alanine_molssi = MolSSI scan (20deg grid, Hartree)
#   the rest       = MMFF94 relaxed scans done here with rdkit (15deg grid)
# everything gets cached to data/ so the scans only run once

from pathlib import Path

import numpy as np
import pandas as pd

HARTREE = 627.509   # -> kcal/mol
MOLSSI_URL = ("https://raw.githubusercontent.com/MolSSI-Education/python-visualization/"
              "main/molssi_python_visualization/alanine-dipeptide/a1a2ene.csv")

# name: smiles, smarts, phi atoms, psi atoms (positions inside the smarts match)
MOLECULES = {
    "pentane":         ("CCCCC", "[CH3][CH2][CH2][CH2][CH3]", (0, 1, 2, 3), (1, 2, 3, 4)),
    "dimethylpentane": ("CC(C)CC(C)C", "[CH3][CH1]([CH3])[CH2][CH1]([CH3])[CH3]", (0, 1, 3, 4), (1, 3, 4, 5)),
    "glycine":         ("CC(=O)NCC(=O)NC", "C(=O)N[CH2]C(=O)N", (0, 2, 3, 4), (2, 3, 4, 6)),
    "alanine_mmff":    ("CC(=O)N[C@@H](C)C(=O)NC", "C(=O)N[CH1](C)C(=O)N", (0, 2, 3, 5), (2, 3, 5, 7)),
    "valine":          ("CC(=O)N[C@@H](C(C)C)C(=O)NC", "C(=O)N[CH1](C(C)C)C(=O)N", (0, 2, 3, 7), (2, 3, 7, 9)),
}
ALL_LANDSCAPES = ["alanine_molssi", *MOLECULES]


def load_molssi(cache_dir):
    path = Path(cache_dir) / "alanine_molssi.csv"
    if path.exists():
        return pd.read_csv(path)
    raw = pd.read_csv(MOLSSI_URL)
    df = raw[(raw.phi < 180) & (raw.psi < 180)].reset_index(drop=True)   # -180 and +180 are the same point
    df["E_kcal"] = (df.energy - df.energy.min()) * HARTREE
    df = df[["phi", "psi", "E_kcal"]]
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


# ---------------- MMFF scan ----------------
# first version relaxed every grid point from ONE start geometry -> loads of points stuck in
# high-energy local minima (pentane maxed at 27 kcal/mol instead of 18, nothing could fit it).
# fix: try 5 embedded conformers + the already-relaxed neighbours, keep the lowest, 2 sweeps.

def scan_mmff(smiles, smarts, phi_pos, psi_pos, step=15, nconf=5):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Chem import rdForceFieldHelpers as FFH
    from rdkit.Chem import rdMolTransforms as T

    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    cids = list(AllChem.EmbedMultipleConfs(mol, nconf, randomSeed=1))
    FFH.MMFFOptimizeMoleculeConfs(mol)
    match = mol.GetSubstructMatch(Chem.MolFromSmarts(smarts))
    phi = [match[i] for i in phi_pos]
    psi = [match[i] for i in psi_pos]
    props = FFH.MMFFGetMoleculeProperties(mol)
    starts = [np.array(mol.GetConformer(c).GetPositions()) for c in cids]
    conf = mol.GetConformer(cids[0])
    grid = list(range(-180, 180, step))
    n = len(grid)
    best = {}   # (i, j) -> (energy, coords)

    def relax(pos, a, b):
        for i, p in enumerate(pos):
            conf.SetAtomPosition(i, p.tolist())
        T.SetDihedralDeg(conf, *phi, float(a))
        T.SetDihedralDeg(conf, *psi, float(b))
        ff = FFH.MMFFGetMoleculeForceField(mol, props, confId=cids[0])
        ff.MMFFAddTorsionConstraint(*phi, False, float(a), float(a), 1e4)
        ff.MMFFAddTorsionConstraint(*psi, False, float(b), float(b), 1e4)
        ff.Minimize(maxIts=2000)
        # energy WITHOUT the constraint terms
        e = FFH.MMFFGetMoleculeForceField(mol, props, confId=cids[0]).CalcEnergy()
        return e, np.array(conf.GetPositions())

    for sweep in range(2):
        for i, a in enumerate(grid):
            for j, b in enumerate(grid):
                cands = list(starts) if sweep == 0 else []
                for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)):
                    key = ((i + di) % n, (j + dj) % n)
                    if key in best:
                        cands.append(best[key][1])
                for pos in cands:
                    e, p = relax(pos, a, b)
                    if (i, j) not in best or e < best[(i, j)][0] - 1e-6:
                        best[(i, j)] = (e, p)

    rows = [(grid[i], grid[j], best[(i, j)][0]) for i in range(n) for j in range(n)]
    df = pd.DataFrame(rows, columns=["phi", "psi", "E"])
    df["E_kcal"] = df.E - df.E.min()
    return df[["phi", "psi", "E_kcal"]]


def load_landscape(name, cache_dir):
    if name == "alanine_molssi":
        return load_molssi(cache_dir)
    path = Path(cache_dir) / f"scan_{name}.csv"
    if path.exists():
        return pd.read_csv(path)
    print(f"  scanning {name} (MMFF94, ~1 min, cached after)...", flush=True)
    df = scan_mmff(*MOLECULES[name])
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def arrays(df):
    X = np.deg2rad(df[["phi", "psi"]].to_numpy()).astype(np.float32)
    y = df["E_kcal"].to_numpy().astype(np.float32)
    return X, y


def spread(y):
    return float(np.std(y))   # population std of E - what the noise budget scales with


# sanity check on the data: a plain 2D fourier series (ridge) should fit a properly
# converged relaxed scan well under 1 kcal/mol. if it can't, the scan is junk, not the model
def fourier_baseline_rmse(df, order=6, ridge=1e-2, n_splits=3):
    P = np.deg2rad(df.phi.to_numpy())
    S = np.deg2rad(df.psi.to_numpy())
    y = df.E_kcal.to_numpy()
    cols = [np.ones_like(P)]
    for i in range(order + 1):
        for j in range(order + 1):
            if i == j == 0:
                continue
            for fa in (np.cos, np.sin):
                for fb in (np.cos, np.sin):
                    cols.append(fa(i * P) * fb(j * S))
    F = np.column_stack(cols)
    errs = []
    for seed in range(n_splits):
        idx = np.random.default_rng(seed).permutation(len(y))
        k = int(0.8 * len(y))
        tr, te = idx[:k], idx[k:]
        w = np.linalg.solve(F[tr].T @ F[tr] + ridge * np.eye(F.shape[1]), F[tr].T @ y[tr])
        errs.append(np.sqrt(np.mean((F[te] @ w - y[te]) ** 2)))
    return float(np.mean(errs))
