from pathlib import Path
import numpy as np
import scanpy as sc

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "submission"

early = sc.read_h5ad(ROOT / "data/E7.25.h5ad")
late = sc.read_h5ad(ROOT / "data/E8.0.h5ad")
baseline = sc.read_h5ad(OUT / "e750_validation_v2.h5ad")

rng = np.random.default_rng(42)

def radius(x):
    centered = x - x.mean(axis=0)
    return np.sqrt(np.mean(np.sum(centered**2, axis=1)))

xyz_early = np.asarray(
    early.obsm["spatial_3D"], dtype=np.float64
)
xyz_late = np.asarray(
    late.obsm["spatial_3D"], dtype=np.float64
)

# Predict E7.5 using E7.25 and E8.0 geometry.
alpha = 1 / 3
target_rg = (
    radius(xyz_early) ** (1 - alpha)
    * radius(xyz_late) ** alpha
)

# Sample 4,000 distinct E7.25 locations.
indices = rng.choice(
    early.n_obs, baseline.n_obs, replace=False
)

sample = xyz_early[indices]
center = sample.mean(axis=0)

xyz_pred = (
    center
    + target_rg / radius(sample) * (sample - center)
)

# Geometry-only experiment:
# retain V2 expression so comparison isolates shape.
pred = baseline.copy()
pred.obsm["spatial_3D"] = xyz_pred.astype(np.float32)

path = OUT / "e750_backtest_early_geometry.h5ad"
pred.write_h5ad(path)

print("Saved:", path)
print("Shape:", pred.shape)
print("Radius of gyration:", round(radius(xyz_pred), 2))
print("Unique positions:", len(np.unique(xyz_pred, axis=0)))
