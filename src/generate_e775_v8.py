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
ALPHA = 0.5
rng = np.random.default_rng(SEED)

def dense(x):
    if hasattr(x, "toarray"):
        return x.toarray().astype(np.float32)
    return np.asarray(x, dtype=np.float32)

def radius(x):
    y = x - x.mean(axis=0)
    return np.sqrt(np.mean(np.sum(y**2, axis=1)))

print("Loading E7.5 and E8.0...")

early = sc.read_h5ad(ROOT / "data/E7.5.h5ad")
late = sc.read_h5ad(ROOT / "data/E8.0.h5ad")

# Official final-test gene panel
panel_path = (
    OUT / "T2__embryo__test_interp.genes.txt"
)

if not panel_path.exists():
    raise FileNotFoundError(panel_path)

panel = [
    line.strip()
    for line in panel_path.read_text().splitlines()
    if line.strip()
]

assert len(panel) == 497
assert len(set(panel)) == 497
assert all(g in early.var_names for g in panel)
assert all(g in late.var_names for g in panel)

# Keep each sampled E7.5 cell paired with
# its original expression and location.
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

xyz_sample = xyz_early_all[source_idx].copy()

# Estimate intermediate overall spatial spread.
target_rg = (
    radius(xyz_early_all)**(1 - ALPHA)
    * radius(xyz_late_all)**ALPHA
)

center = xyz_sample.mean(axis=0)
xyz_pred = center + (
    target_rg / radius(xyz_sample)
) * (xyz_sample - center)

print("Target radius:", round(target_rg, 2))
print("Predicted radius:", round(radius(xyz_pred), 2))

# Construct RNA neighbors in a shared feature space.
# All observed E8.0 cells are possible future states.
print("Finding RNA-matched E8.0 cells...")

joint = np.vstack([Xearly, Xlate])

scaled = StandardScaler().fit_transform(joint)

Z = PCA(
    n_components=30,
    svd_solver="randomized",
    random_state=SEED
).fit_transform(scaled)

Zearly = Z[:N]
Zlate = Z[N:]

dist, indices = NearestNeighbors(
    n_neighbors=5
).fit(Zlate).kneighbors(Zearly)

bandwidth = max(
    float(np.median(dist[:, 0])), 1e-6
)

log_weights = -(dist**2) / (2 * bandwidth**2)
log_weights -= log_weights.max(axis=1, keepdims=True)

weights = np.exp(log_weights)
weights /= weights.sum(axis=1, keepdims=True)

# Sample one plausible future cell per E7.5 source.
u = rng.random(N)

choice = (
    u[:, None] > np.cumsum(weights, axis=1)
).sum(axis=1)

choice = np.clip(choice, 0, 4)
future_indices = indices[np.arange(N), choice]

Xfuture = Xlate[future_indices]

# Use independent RNGs for reproducible comparisons.
for fraction in [0.0, 0.25, 0.375]:
    local_rng = np.random.default_rng(
        SEED + round(1000 * fraction)
    )

    # Preserve early observed profiles except for
    # selected cells that acquire a matched late profile.
    choose_late = local_rng.random(N) < fraction

    Xpred = Xearly.copy()
    Xpred[choose_late] = Xfuture[choose_late]

    name = f"e775_v8_early_mix_{round(100*fraction):02d}"
    path = OUT / f"{name}.h5ad"

    pred = ad.AnnData(
        X=Xpred.astype(np.float32),
        obs=pd.DataFrame(
            index=[f"v8_{i:05d}" for i in range(N)]
        ),
        var=pd.DataFrame(index=panel)
    )

    pred.obsm["spatial_3D"] = xyz_pred.astype(np.float32)

    assert pred.shape == (N, 497)
    assert list(pred.var_names) == panel
    assert np.isfinite(pred.X).all()
    assert (pred.X >= 0).all()
    assert np.isfinite(pred.obsm["spatial_3D"]).all()
    assert len(np.unique(
        pred.obsm["spatial_3D"], axis=0
    )) == N

    pred.write_h5ad(path)

    print(f"\n{name}")
    print("  Shape:", pred.shape)
    print("  Actual late fraction:", round(choose_late.mean(), 4))
    print("  Fraction zeros:", round(float((pred.X == 0).mean()), 4))
    print("  Radius:", round(radius(xyz_pred), 2))
    print("  Saved:", path)

print("\nV8 candidate generation complete.")
