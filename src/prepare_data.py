from pathlib import Path
import numpy as np
import scanpy as sc
import anndata as ad
from scipy import sparse

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

stages = {
    "E6.75": 6.75,
    "E7.25": 7.25,
    "E8.0": 8.0,
}

rng = np.random.default_rng(42)
datasets = {}

for stage, time in stages.items():
    adata = sc.read_h5ad(DATA / f"{stage}.h5ad")

    # Use only genes shared across stages
    if stage == "E6.75":
        common = set(adata.var_names)
    else:
        common &= set(adata.var_names)

    datasets[stage] = adata

genes = sorted(common)
print("Shared genes:", len(genes))

samples = []

for stage, time in stages.items():
    adata = datasets[stage][:, genes].copy()

    # Use the existing log1p layer as a candidate
    # expression representation
    adata.X = adata.layers["log1p"].copy()

    # Sample cells for a manageable pilot
    n = min(1000, adata.n_obs)
    idx = rng.choice(adata.n_obs, size=n, replace=False)
    adata = adata[idx].copy()

    adata.obs["time"] = time
    adata.obs["stage"] = stage
    adata.obs_names = [
        f"{stage}_{i}" for i in range(adata.n_obs)
    ]

    samples.append(adata)

combined = ad.concat(
    samples,
    join="inner",
    merge="same",
    uns_merge="same",
)

# Preserve original 3D coordinates
assert "spatial_3D" in combined.obsm

# PCA on the expression representation
sc.pp.pca(combined, n_comps=30)

output = RESULTS / "embryo_pilot.h5ad"
combined.write_h5ad(output)

print("Saved:", output)
print("Shape:", combined.shape)
print("PCA:", combined.obsm["X_pca"].shape)
print("Spatial:", combined.obsm["spatial_3D"].shape)
print(combined.obs["stage"].value_counts())
