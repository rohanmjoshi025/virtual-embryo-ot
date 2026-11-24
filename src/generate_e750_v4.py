"""E7.5 Task 2 validation: controlled expression experiments.

Requirements: numpy, scipy, pandas, scanpy, anndata, scikit-learn.
Runs from any working directory when saved in project/src/.
Expected project files:
    data/E7.25.h5ad
    data/E8.0.h5ad
    submission/e750_validation_v2.h5ad

Writes three candidates sharing V2's *exact* geometry and cell order.
No E7.5 truth is used in prediction generation.
"""

from pathlib import Path

import numpy as np
import scanpy as sc
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "submission"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
ALPHA = 1 / 3  # E7.5 between E7.25 and E8.0
TARGET_SUM = 10_000.0
SPATIAL_SCALE = (
    147.29 ** (1 - ALPHA) * 326.56 ** ALPHA / 326.56
)


def dense(X):
    if hasattr(X, "toarray"):
        X = X.toarray()
    return np.asarray(X, dtype=np.float32)


def linear_normalized(X_log):
    """Invert log1p and normalize each cell to 10,000 linear units."""
    X_lin = np.expm1(np.clip(X_log.astype(np.float64), 0, 20))
    sums = X_lin.sum(axis=1, keepdims=True)
    if np.any(sums <= 0):
        raise ValueError("Found a cell with zero total expression")
    X_lin *= TARGET_SUM / sums
    return X_lin.astype(np.float32)


def normalized_log(X_lin):
    """Return log1p expression normalized to a 10,000-unit library."""
    X_lin = np.maximum(np.asarray(X_lin, dtype=np.float64), 0)
    sums = X_lin.sum(axis=1, keepdims=True)
    if np.any(sums <= 0):
        raise ValueError("Predicted a zero-library cell")
    X_lin *= TARGET_SUM / sums
    return np.log1p(X_lin).astype(np.float32)


def save_candidate(v2, x_pred, name):
    pred = v2.copy()
    pred.X = np.asarray(x_pred, dtype=np.float32)
    # Keep V2's coordinates, ordering, and gene panel completely unchanged.
    assert pred.shape == (v2.n_obs, 498)
    assert np.array_equal(pred.obsm["spatial_3D"],
                          v2.obsm["spatial_3D"])
    assert np.isfinite(pred.X).all()
    assert (pred.X >= 0).all()

    filename = OUT / f"e750_{name}.h5ad"
    pred.write_h5ad(filename)

    sums = np.expm1(pred.X.astype(np.float64)).sum(axis=1)
    print(f"\n{name}: {filename}")
    print("  Cells / genes:", pred.shape)
    print("  Median implied library:", round(float(np.median(sums)), 2))
    print("  Fraction zero:", round(float(np.mean(pred.X == 0)), 4))
    print("  Mean per-gene variance:",
          round(float(pred.X.var(axis=0).mean()), 4))


print("Loading E7.25, E8.0, and saved E7.5 V2...")
a = sc.read_h5ad(ROOT / "data" / "E7.25.h5ad")
b = sc.read_h5ad(ROOT / "data" / "E8.0.h5ad")
v2 = sc.read_h5ad(OUT / "e750_validation_v2.h5ad")

panel = list(v2.var_names)
assert len(panel) == 498
assert all(g in a.var_names for g in panel)
assert all(g in b.var_names for g in panel)

Xa = dense(a[:, panel].X)
Xb = dense(b[:, panel].X)
n_pred = v2.n_obs

# The original V2 draws its E8.0 donors as the very first RNG operation.
rng = np.random.default_rng(SEED)
target_idx = rng.choice(b.n_obs, size=n_pred, replace=False)
X_target = Xb[target_idx]

# Reconstruct the original donor geometry and verify row alignment.
xyz_target = np.asarray(b.obsm["spatial_3D"],
                        dtype=np.float64)[target_idx]
xyz_center = xyz_target.mean(axis=0)
xyz_reconstructed = (
    xyz_center + SPATIAL_SCALE * (xyz_target - xyz_center)
).astype(np.float32)
if not np.allclose(xyz_reconstructed, v2.obsm["spatial_3D"],
                   rtol=1e-5, atol=1e-3):
    raise RuntimeError(
        "Could not reproduce V2 cell order / geometry. "
        "Check the original seed, scale, and sampling code."
    )
print("V2 donor cells and coordinate order verified.")

# Reproduce V2's RNA-neighbor matching in one shared PCA basis.
print("Computing RNA nearest neighbors...")
Z = PCA(n_components=30, random_state=SEED).fit_transform(
    StandardScaler().fit_transform(np.vstack([Xa, X_target]))
)
Za, Zb = Z[:a.n_obs], Z[a.n_obs:]
knn = NearestNeighbors(n_neighbors=5)
knn.fit(Za)
distances, indices = knn.kneighbors(Zb)
bandwidth = max(float(np.median(distances[:, 0])), 1e-6)
weights = np.exp(-(distances ** 2) / (2 * bandwidth ** 2))
weights /= weights.sum(axis=1, keepdims=True)

# Check how closely our re-created expression matches V2.
early_soft_log = (Xa[indices] * weights[:, :, None]).sum(axis=1)
x_v2_reconstructed = (
    (1 - ALPHA) * early_soft_log + ALPHA * X_target
)
print("Max reconstructed V2 expression difference:",
      round(float(np.max(np.abs(x_v2_reconstructed - dense(v2.X)))), 5))

# A: change ONLY normalization of original V2 expression.
x_renorm = normalized_log(linear_normalized(dense(v2.X)))
save_candidate(v2, x_renorm, "v4_renorm")

# B: interpolate in linear expression space; soft 5-neighbor matching.
Xa_linear = linear_normalized(Xa)
X_target_linear = linear_normalized(X_target)
early_soft_linear = (
    Xa_linear[indices] * weights[:, :, None]
).sum(axis=1)
x_soft = normalized_log(
    (1 - ALPHA) * early_soft_linear + ALPHA * X_target_linear
)
save_candidate(v2, x_soft, "v4_soft_linear")

# C: choose ONE earlier-stage cell from the five RNA-nearest neighbors.
# This avoids averaging five log profiles or five early cell profiles.
draw_rng = np.random.default_rng(SEED + 101)
choice = (draw_rng.random(n_pred)[:, None]
          > np.cumsum(weights, axis=1)).sum(axis=1)
choice = np.minimum(choice, weights.shape[1] - 1)
source_idx = indices[np.arange(n_pred), choice]
early_hard_linear = Xa_linear[source_idx]
x_hard = normalized_log(
    (1 - ALPHA) * early_hard_linear + ALPHA * X_target_linear
)
save_candidate(v2, x_hard, "v4_hard_linear")

print("\nGenerated 3 V4 candidates. None used E7.5 validation truth.")
print("Next: score all candidates using the released E7.5 ground truth.")
