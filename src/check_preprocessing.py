from pathlib import Path
import numpy as np
import scanpy as sc
from scipy import sparse

data_dir = Path(__file__).resolve().parent.parent / "data"
stages = ["E6.75", "E7.25", "E8.0"]

datasets = {
    stage: sc.read_h5ad(data_dir / f"{stage}.h5ad")
    for stage in stages
}

# 1. Check common genes
gene_sets = {
    stage: set(adata.var_names)
    for stage, adata in datasets.items()
}

common_genes = set.intersection(*gene_sets.values())

print("Shared genes:", len(common_genes))

for stage, genes in gene_sets.items():
    print(
        stage,
        "genes not shared:",
        sorted(genes - common_genes)
    )

# 2. Check expression distributions
for stage, adata in datasets.items():
    X = adata.X

    values = X.data if sparse.issparse(X) else np.asarray(X).ravel()

    print(f"\n{stage}")
    print("Matrix type:", type(X))
    print("Minimum:", float(values.min()))
    print("Maximum:", float(values.max()))
    print("Mean:", float(X.mean()))
    print("Non-integer values:", bool(
        np.any(np.abs(values - np.round(values)) > 1e-5)
    ))

    print("Layers:", list(adata.layers.keys()))
    print("Unstructured metadata:", list(adata.uns.keys()))

# 3. Check spatial coordinate ranges
for stage, adata in datasets.items():
    coords = np.asarray(adata.obsm["spatial_3D"])

    print(f"\n{stage} coordinate ranges:")
    for axis in range(3):
        print(
            f"Axis {axis}:",
            coords[:, axis].min(),
            coords[:, axis].max()
        )

# 4. Check cell-type labels
for stage, adata in datasets.items():
    print(f"\n{stage} cell types:")
    print(adata.obs["celltype"].value_counts().head(10))