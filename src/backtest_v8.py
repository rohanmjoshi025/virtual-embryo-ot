from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "submission"

SEED = 42
N = 4000
ALPHA = 1 / 3

rng = np.random.default_rng(SEED)

def dense(x):
    if hasattr(x, "toarray"):
        return x.toarray().astype(np.float32)
    return np.asarray(x, dtype=np.float32)

def radius(x):
    centered = x - x.mean(axis=0)
    return np.sqrt(np.mean(np.sum(centered**2, axis=1)))

print("Loading E7.25 and E8.0...")

early = sc.read_h5ad(ROOT / "data/E7.25.h5ad")
late = sc.read_h5ad(ROOT / "data/E8.0.h5ad")

# Use the accepted E7.5 validation panel.
panel = list(early.var_names)
assert len(panel) == 498

# Preserve observed early-stage cell pairs.
source_idx = rng.choice(
    early.n_obs, N, replace=False
)

Xearly = dense(early[source_idx, panel].X)
Xlate = dense(late[:, panel].X)

xyz_early_all = np.asarray(
    early.obsm["spatial_3D"], dtype=np.float64
)

xyz_late_all = np.asarray(
    late.obsm["spatial_3D"], dtype=np.float64
)

xyz = xyz_early_all[source_idx].copy()

# Predict intermediate spatial spread.
target_rg = (
    radius(xyz_early_all)**(1 - ALPHA)
    * radius(xyz_late_all)**ALPHA
)

center = xyz.mean(axis=0)

xyz_pred = (
    center
    + target_rg / radius(xyz) * (xyz - center)
).astype(np.float32)

print("Predicted radius:", round(radius(xyz_pred), 2))

# Match early-stage cells to plausible E8.0 states.
print("Computing RNA neighbors...")

joint = np.vstack([Xearly, Xlate])
joint = StandardScaler().fit_transform(joint)

Z = PCA(
    n_components=30,
    svd_solver="randomized",
    random_state=SEED
).fit_transform(joint)

Zearly = Z[:N]
Zlate = Z[N:]

dist, indices = NearestNeighbors(
    n_neighbors=5
).fit(Zlate).kneighbors(Zearly)

bandwidth = max(float(np.median(dist[:, 0])), 1e-6)

log_weights = -dist**2 / (2 * bandwidth**2)
log_weights -= log_weights.max(axis=1, keepdims=True)

weights = np.exp(log_weights)
weights /= weights.sum(axis=1, keepdims=True)

u = rng.random(N)

choice = (
    u[:, None] > np.cumsum(weights, axis=1)
).sum(axis=1)

choice = np.clip(choice, 0, 4)

future_idx = indices[np.arange(N), choice]
Xfuture = Xlate[future_idx]

for fraction in [0.0, 0.25, 0.375, 0.50]:

    local_rng = np.random.default_rng(
        SEED + round(1000 * fraction)
    )

    choose_late = local_rng.random(N) < fraction

    Xpred = Xearly.copy()
    Xpred[choose_late] = Xfuture[choose_late]

    name = f"e750_v8_backtest_{round(100*fraction):02d}"

    pred = ad.AnnData(
        X=Xpred.astype(np.float32),
        obs=pd.DataFrame(
            index=[f"v8test_{i:05d}" for i in range(N)]
        ),
        var=pd.DataFrame(index=panel)
    )

    pred.obsm["spatial_3D"] = xyz_pred.copy()

    assert pred.shape == (4000, 498)
    assert np.isfinite(pred.X).all()
    assert np.isfinite(pred.obsm["spatial_3D"]).all()

    output = OUT / f"{name}.h5ad"
    pred.write_h5ad(output)

    print(f"\n{name}")
    print("  Late fraction:", round(choose_late.mean(), 4))
    print("  Expression zeros:", round(
        float((Xpred == 0).mean()), 4
    ))
    print("  Saved:", output)

print("\nV8 backtest generation complete.")
