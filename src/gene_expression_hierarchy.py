
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

adata = sc.read_h5ad(RESULTS / "embryo_pilot.h5ad")

# Use gene expression, not PCA coordinates
# Group by both stage and annotated cell type
adata.obs["stage_celltype"] = (
    adata.obs["time"].astype(str)
    + " | "
    + adata.obs["celltype"].astype(str)
)

# Mean expression per stage and cell type
df = adata.to_df()

mean_expression = df.groupby(
    adata.obs["stage_celltype"],
    observed=True
).mean()

# Remove genes with no expression variation across groups
gene_variance = mean_expression.var(axis=0)
mean_expression = mean_expression.loc[
    :, gene_variance > 0
]

# Standardize each gene across groups for visualization
gene_mean = mean_expression.mean(axis=0)
gene_std = mean_expression.std(axis=0)

scaled = (mean_expression - gene_mean) / gene_std

# Cluster stage-cell-type populations by expression
cell_dist = pdist(scaled.values, metric="euclidean")
cell_linkage = linkage(cell_dist, method="average")

# Cluster genes by their patterns across populations
gene_dist = pdist(scaled.values.T, metric="euclidean")
gene_linkage = linkage(gene_dist, method="average")

# Plot the cell-population dendrogram
plt.figure(figsize=(15, 7))

dendrogram(
    cell_linkage,
    labels=scaled.index.tolist(),
    leaf_rotation=90,
    leaf_font_size=7
)

plt.title("Embryonic Cell-Population Expression Hierarchy")
plt.ylabel("Expression-profile distance")
plt.tight_layout()

plt.savefig(
    RESULTS / "celltype_expression_dendrogram.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# Save mean expression for later gene-specificity analysis
mean_expression.to_csv(
    RESULTS / "mean_expression_by_stage_celltype.csv"
)

print("Saved dendrogram and mean-expression matrix.")
