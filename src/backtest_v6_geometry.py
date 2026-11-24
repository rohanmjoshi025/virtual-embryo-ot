from pathlib import Path
import numpy as np
import scanpy as sc

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "submission"
ALPHA = 1 / 3

early = sc.read_h5ad(ROOT / "data/E7.25.h5ad")
late = sc.read_h5ad(ROOT / "data/E8.0.h5ad")
baseline = sc.read_h5ad(
    OUT / "e750_validation_v2.h5ad"
)

def radius(xyz):
    centered = xyz - xyz.mean(axis=0)
    return np.sqrt(np.mean(np.sum(centered**2, axis=1)))

def axis_std(xyz):
    values = np.linalg.eigvalsh(
        np.cov(xyz, rowvar=False)
    )
    return np.sqrt(np.maximum(values[::-1], 0))

def reshape_pca(xyz, desired):
    center = xyz.mean(axis=0)
    values, vectors = np.linalg.eigh(
        np.cov(xyz, rowvar=False)
    )
    order = np.argsort(values)[::-1]
    basis = vectors[:, order]

    projected = (xyz - center) @ basis
    current = projected.std(axis=0, ddof=1)

    projected *= desired / current

    return projected @ basis.T + center

xyz_a = np.asarray(
    early.obsm["spatial_3D"], dtype=np.float64
)
xyz_b_full = np.asarray(
    late.obsm["spatial_3D"], dtype=np.float64
)

# Recover the E8.0 donor selection used by V2.
rng = np.random.default_rng(42)
indices = rng.choice(
    late.n_obs,
    baseline.n_obs,
    replace=False
)

xyz_b = xyz_b_full[indices]

# Verify donor selection reproduces V2 positions.
v2_scale = 0.5881302486091871
center = xyz_b.mean(axis=0)
reconstructed = center + v2_scale * (xyz_b - center)

assert np.allclose(
    reconstructed,
    baseline.obsm["spatial_3D"],
    atol=0.02
), "V2 donor selection does not match!"

# Candidate A: PCA shape interpolation.
desired_std = (
    (1 - ALPHA) * axis_std(xyz_a)
    + ALPHA * axis_std(xyz_b_full)
)

xyz_pca = reshape_pca(xyz_b, desired_std)

# Candidate B: isotropic size interpolation.
rg_goal = (
    radius(xyz_a)**(1 - ALPHA)
    * radius(xyz_b_full)**ALPHA
)

xyz_iso = center + (
    rg_goal / radius(xyz_b)
) * (xyz_b - center)

for name, xyz in [
    ("e750_backtest_pca", xyz_pca),
    ("e750_backtest_iso", xyz_iso),
]:
    pred = baseline.copy()
    pred.obsm["spatial_3D"] = xyz.astype(np.float32)

    path = OUT / f"{name}.h5ad"
    pred.write_h5ad(path)

    print(name)
    print("  Shape:", pred.shape)
    print("  Rg:", round(float(radius(xyz)), 2))
    print("  Axis std:", np.round(axis_std(xyz), 2))
    print("  Saved:", path)

print("\nBacktest candidates generated.")
