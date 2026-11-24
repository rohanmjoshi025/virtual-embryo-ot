"""V7 E7.75 expression-mixture candidates for the T2 embryo final board.

Requires:
  data/E7.5.h5ad
  data/E8.0.h5ad
  submission/e775_v6_iso_soft.h5ad
  submission/T2__embryo__test_interp.genes.txt  (official downloaded panel)

Run from any directory: python /path/to/project/src/generate_e775_v7.py
"""
from pathlib import Path

import numpy as np
import scanpy as sc
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "submission"
SEED = 42
N = 4000


def dense(matrix):
    return (matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)).astype(np.float32)


def radius(xyz):
    xyz = np.asarray(xyz, dtype=np.float64)
    centered = xyz - xyz.mean(axis=0)
    return float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))


panel_file = OUT / "T2__embryo__test_interp.genes.txt"
if not panel_file.is_file():
    raise FileNotFoundError(
        "Official final panel is required at: " + str(panel_file) + "\n"
        "Download it in your signed-in browser from "
        "https://virtualembryo.ai/challenge/panels/T2__embryo__test_interp.genes.txt"
    )

panel = [g.strip().lstrip("\ufeff") for g in panel_file.read_text(encoding="utf-8").splitlines() if g.strip()]
if len(panel) != 497 or len(set(panel)) != 497:
    raise ValueError(f"Expected exactly 497 unique genes in official panel, got {len(panel)}")

print("Loading E7.5 and E8.0, plus saved V6 isotropic geometry...")
early = sc.read_h5ad(DATA / "E7.5.h5ad")
late = sc.read_h5ad(DATA / "E8.0.h5ad")
v6 = sc.read_h5ad(OUT / "e775_v6_iso_soft.h5ad")

for name, data in [("E7.5", early), ("E8.0", late), ("V6", v6)]:
    missing = sorted(set(panel) - set(data.var_names))
    if missing:
        raise ValueError(f"{name} lacks genes from the official panel: {missing[:10]}")
if v6.n_obs != N:
    raise ValueError(f"Expected {N} V6 cells, received {v6.n_obs}")

# Reconstruct the E8.0 donor order used by the original V6 generator.
late_idx = np.random.default_rng(SEED).choice(late.n_obs, N, replace=False)
xyz_late = np.asarray(late.obsm["spatial_3D"], dtype=np.float64)[late_idx]
rg_target = np.sqrt(radius(early.obsm["spatial_3D"]) * radius(late.obsm["spatial_3D"]))
center = xyz_late.mean(axis=0)
xyz_expected = center + (rg_target / radius(xyz_late)) * (xyz_late - center)
if not np.allclose(v6.obsm["spatial_3D"], xyz_expected, atol=0.02):
    raise RuntimeError("V6 donor coordinates do not match the reconstructed E8.0 samples")
print("V6 donor positions and order verified.")

# Use exactly the same official genes for embedding, prediction and output.
Xa = dense(early[:, panel].X)
Xb = dense(late[:, panel].X)[late_idx]

print("Fitting 30-dimensional RNA PCA and matching five E7.5 neighbors...")
features = np.vstack((Xa, Xb))
features = StandardScaler().fit_transform(features)
Z = PCA(n_components=30, random_state=SEED).fit_transform(features)
Zearly = Z[:early.n_obs]
Zlate = Z[early.n_obs:]

dist, idx = NearestNeighbors(n_neighbors=5).fit(Zearly).kneighbors(Zlate)
sigma = max(float(np.median(dist[:, 0])), 1e-6)
energy = dist**2
weights = np.exp(-(energy - energy.min(axis=1, keepdims=True)) / (2 * sigma**2))
weights /= weights.sum(axis=1, keepdims=True)

match_rng = np.random.default_rng(123)
choices = (match_rng.random(N)[:, None] > np.cumsum(weights, axis=1)).sum(axis=1)
choices = np.clip(choices, 0, idx.shape[1] - 1)
selected_early = idx[np.arange(N), choices]
Xearly_sampled = Xa[selected_early]

# Use one common random number per cell so all fractions can be compared fairly.
mixture_uniform = np.random.default_rng(2026).random(N)

for fraction in (0.25, 0.375, 0.50):
    choose_late = mixture_uniform < fraction
    Xmix = Xearly_sampled.copy()
    Xmix[choose_late] = Xb[choose_late]

    prediction = v6[:, panel].copy()
    prediction.X = Xmix.astype(np.float32)
    # A final T2 submission needs expression + spatial coordinates, no cell-type labels.
    prediction.obs = prediction.obs.iloc[:, 0:0].copy()

    if prediction.shape != (N, 497):
        raise ValueError(f"Unexpected prediction shape: {prediction.shape}")
    if not np.isfinite(prediction.X).all() or (prediction.X < 0).any():
        raise ValueError("Invalid predicted expression values")
    if not np.isfinite(prediction.obsm["spatial_3D"]).all():
        raise ValueError("Invalid predicted coordinates")
    if list(prediction.var_names) != panel:
        raise ValueError("Official gene order was not retained")

    label = int(round(fraction * 100))
    path = OUT / f"e775_v7_mix_{label:02d}_iso.h5ad"
    prediction.write_h5ad(path)
    check = sc.read_h5ad(path)
    assert list(check.var_names) == panel and check.shape == (N, 497)

    library = np.expm1(Xmix).sum(axis=1)
    print(f"\n{path.name}")
    print("  Actual late-stage fraction:", round(float(choose_late.mean()), 4))
    print("  Shape:", prediction.shape)
    print("  Median implied library size:", round(float(np.median(library)), 1))
    print("  Fraction of zero-expression values:", round(float(np.mean(Xmix == 0)), 4))
    print("  Radius of gyration:", round(radius(prediction.obsm["spatial_3D"]), 2))
    print("  Unique spatial positions:", len(np.unique(prediction.obsm["spatial_3D"], axis=0)))

print("\nV7 files are ready for official FORMAT checks, not yet validated or scored by the challenge.")
