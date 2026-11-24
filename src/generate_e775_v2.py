
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "submission"
OUT.mkdir(exist_ok=True)

SEED = 42
ALPHA = 2 / 3
SPATIAL_SCALE = 0.75

rng = np.random.default_rng(SEED)

def dense(x):
    if hasattr(x, "toarray"):
        return x.toarray().astype(np.float32)
    return np.asarray(x, dtype=np.float32)

print("Loading observed stages...")

a = sc.read_h5ad(ROOT / "data/E7.25.h5ad")
b = sc.read_h5ad(ROOT / "data/E8.0.h5ad")

# Use the 498-gene panel that passed validation.
panel = list(a.var_names)

assert len(panel) == 498
assert all(g in b.var_names for g in panel)

Xa = dense(a[:, panel].X)
Xb = dense(b[:, panel].X)

# Estimate the E7.75 cell count.
n_pred = round(
    a.n_obs + ALPHA * (b.n_obs - a.n_obs)
)

print("Target cell count:", n_pred)

# Sample E8.0 cells WITHOUT replacement.
target_idx = rng.choice(
    b.n_obs,
    size=n_pred,
    replace=False
)

X_target = Xb[target_idx]

# Learn a common RNA embedding.
print("Building RNA PCA embedding...")

X_joint = np.vstack([Xa, X_target])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_joint)

pca = PCA(
    n_components=30,
    random_state=SEED
)

Z = pca.fit_transform(X_scaled)

Za = Z[:a.n_obs]
Zb = Z[a.n_obs:]

# Match each selected E8.0 cell to a nearby
# E7.25 expression profile.
print("Matching cell states...")

nn = NearestNeighbors(
    n_neighbors=5,
    algorithm="auto"
)

nn.fit(Za)

distances, indices = nn.kneighbors(Zb)

# Soft weights for the five RNA neighbors.
# The bandwidth is estimated from the data.
bandwidth = max(
    float(np.median(distances[:, 0])),
    1e-6
)

weights = np.exp(
    -(distances ** 2) / (2 * bandwidth ** 2)
)

weights /= weights.sum(axis=1, keepdims=True)

# Estimate earlier-stage expression.
X_early = np.sum(
    Xa[indices] * weights[:, :, None],
    axis=1
)

# Interpolate expression.
X_pred = (
    (1 - ALPHA) * X_early
    + ALPHA * X_target
)

X_pred = np.maximum(
    X_pred, 0
).astype(np.float32)

print("Generating spatial coordinates...")

# Preserve E8.0 spatial structure while
# testing a smaller intermediate scale.
xyz = np.asarray(
    b.obsm["spatial_3D"],
    dtype=np.float32
)[target_idx].copy()

center = xyz.mean(axis=0)

xyz_pred = (
    center + SPATIAL_SCALE * (xyz - center)
).astype(np.float32)

# Create the submission.
pred = ad.AnnData(
    X=X_pred,
    obs=pd.DataFrame(
        index=[
            f"v2_{i:06d}"
            for i in range(n_pred)
        ]
    ),
    var=pd.DataFrame(index=panel)
)

pred.obsm["spatial_3D"] = xyz_pred

# Validation.
assert pred.shape == (n_pred, 498)
assert xyz_pred.shape == (n_pred, 3)
assert np.isfinite(X_pred).all()
assert np.isfinite(xyz_pred).all()
assert (X_pred >= 0).all()

unique_positions = len(
    np.unique(xyz_pred, axis=0)
)

print("Unique positions:", unique_positions)
print("Spatial extent:",
      np.ptp(xyz_pred, axis=0))

output = OUT / "e775_v2_geometry.h5ad"
pred.write_h5ad(output)

print("\nSaved:", output)
print("Shape:", pred.shape)
print("Done.")
