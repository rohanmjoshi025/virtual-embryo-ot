
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.special import logsumexp
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "submission"
OUT.mkdir(exist_ok=True)

SEED = 42
N = 3000
EPSILON = 0.05
ALPHA = 2 / 3

rng = np.random.default_rng(SEED)

print("Loading data...")
a = sc.read_h5ad(DATA / "E7.25.h5ad")
b = sc.read_h5ad(DATA / "E8.0.h5ad")

# E8.0 provides our candidate 500-gene panel.
panel = list(b.var_names)

# Match genes by name, not by column position.
shared = [g for g in panel if g in a.var_names]

assert len(panel) == 500
assert len(shared) == 498

# Reproducible cell sampling.
ia = rng.choice(a.n_obs, N, replace=False)
ib = rng.choice(b.n_obs, N, replace=False)

a = a[ia].copy()
b = b[ib].copy()

# Dense matrices are manageable for this pilot.
# Preserve the original .X expression representation.
def dense(x):
    return (
        x.toarray().astype(np.float32)
        if hasattr(x, "toarray")
        else np.asarray(x, dtype=np.float32)
    )

Xa = dense(a[:, shared].X)
Xb = dense(b[:, shared].X)

# Joint RNA embedding.
X_joint = np.vstack([Xa, Xb])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_joint)

pca = PCA(n_components=30, random_state=SEED)
Z = pca.fit_transform(X_scaled)

Za = Z[:N]
Zb = Z[N:]

# Scale the embedding so epsilon has a defined
# meaning relative to the observed pairwise costs.
cost = cdist(Za, Zb, metric="sqeuclidean")
cost /= max(float(np.median(cost)), 1e-8)

print("Computing entropic OT coupling...")

# Balanced entropic OT with uniform marginals.
# Log-domain Sinkhorn improves numerical stability.
logK = -cost / EPSILON

log_a = np.full(N, -np.log(N))
log_b = np.full(N, -np.log(N))

u = np.zeros(N)
v = np.zeros(N)

for iteration in range(2000):
    new_u = log_a - logsumexp(logK + v[None, :], axis=1)
    new_v = log_b - logsumexp(
        logK + new_u[:, None], axis=0
    )

    delta = max(
        np.max(np.abs(new_u - u)),
        np.max(np.abs(new_v - v))
    )

    u, v = new_u, new_v

    if delta < 1e-7:
        print("Sinkhorn converged at", iteration + 1)
        break
else:
    print("Warning: Sinkhorn reached iteration limit.")

logP = logK + u[:, None] + v[None, :]

# Row-normalized transport weights.
logQ = logP - logsumexp(logP, axis=1, keepdims=True)
Q = np.exp(logQ)

print("Generating E7.75 expression...")

# All 500 E8.0 genes.
Xb_full = dense(b[:, panel].X)

# Sample target indices once, using OT probabilities.
target_indices = np.array([
    rng.choice(N, p=Q[i] / Q[i].sum())
    for i in range(N)
])

# Use the same target cell for expression and position.
X_future = Xb_full[target_indices]

# Initialize missing E7.25 genes with their
# transport-weighted future estimates.
Xa_full = X_future.copy()

# Replace shared-gene columns with observed E7.25.
shared_index = {
    gene: k for k, gene in enumerate(shared)
}

for j, gene in enumerate(panel):
    if gene in shared_index:
        Xa_full[:, j] = Xa[:, shared_index[gene]]

X_pred = (
    (1 - ALPHA) * Xa_full
    + ALPHA * X_future
)

X_pred = np.maximum(X_pred, 0).astype(np.float32)

print("Generating provisional spatial coordinates...")

# Get E8.0 spatial coordinates
xyz_b = np.asarray(
    b.obsm["spatial_3D"],
    dtype=np.float32
)

# Use the SAME target indices sampled earlier
xyz_pred = xyz_b[target_indices].copy()

print("Unique target positions:",
      len(np.unique(target_indices)))

print("Predicted spatial extent:",
      np.ptp(xyz_pred, axis=0))

prediction = ad.AnnData(
    X=X_pred,
    obs=pd.DataFrame(
        index=[f"pred_{i:05d}" for i in range(N)]
    ),
    var=pd.DataFrame(index=panel)
)

prediction.obsm["spatial_3D"] = xyz_pred

assert prediction.shape == (N, 500)
assert prediction.obsm["spatial_3D"].shape == (N, 3)
assert np.isfinite(prediction.X).all()
assert np.isfinite(xyz_pred).all()
assert (prediction.X >= 0).all()

path = OUT / "e775_ot_baseline.h5ad"
prediction.write_h5ad(path)

print("\nSaved:", path)
print("Shape:", prediction.shape)
print("Spatial shape:", xyz_pred.shape)
print("Expression range:",
      float(X_pred.min()),
      float(X_pred.max()))

print("Done.")
