
from pathlib import Path
import numpy as np
import scanpy as sc
from scipy import sparse

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

stages = ["E6.75", "E7.25", "E8.0"]
datasets = {}

for stage in stages:
    print("\n" + "=" * 60)
    print(stage)
    print("=" * 60)

    adata = sc.read_h5ad(DATA / f"{stage}.h5ad")
    datasets[stage] = adata

    print("Shape:", adata.shape)
    print("First 10 genes:", list(adata.var_names[:10]))
    print("Last 10 genes:", list(adata.var_names[-10:]))

    print("Expression layers:", list(adata.layers.keys()))
    print("Observation columns:", list(adata.obs.columns))
    print("Spatial keys:", list(adata.obsm.keys()))

    for name, matrix in [("X", adata.X)] + [
        (key, adata.layers[key])
        for key in adata.layers.keys()
    ]:
        if sparse.issparse(matrix):
            values = matrix.data
        else:
            values = np.asarray(matrix).ravel()

        if values.size:
            print(
                f"{name}: min={values.min():.4f}, "
                f"max={values.max():.4f}, "
                f"mean_stored={values.mean():.4f}"
            )

    if "spatial_3D" in adata.obsm:
        xyz = np.asarray(adata.obsm["spatial_3D"])
        print("Spatial shape:", xyz.shape)
        print("Spatial minimum:", xyz.min(axis=0))
        print("Spatial maximum:", xyz.max(axis=0))

print("\n" + "=" * 60)
print("GENE PANEL COMPARISON")
print("=" * 60)

gene_sets = {
    stage: set(adata.var_names)
    for stage, adata in datasets.items()
}

common = set.intersection(*gene_sets.values())

print("Genes shared across all stages:", len(common))

for stage in stages:
    unique = gene_sets[stage] - common
    print(f"{stage} non-shared genes:", sorted(unique))

print(
    "E7.25 and E8.0 identical gene order:",
    list(datasets["E7.25"].var_names)
    == list(datasets["E8.0"].var_names)
)

print("\nInspection complete.")
