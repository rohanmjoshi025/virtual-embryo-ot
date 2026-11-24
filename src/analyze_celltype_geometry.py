
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "celltype_geometry"
OUT.mkdir(parents=True, exist_ok=True)

STAGES = ["E7.25", "E8.0"]

records = []
celltype_sets = {}

for stage in STAGES:
    print(f"\nAnalyzing {stage}...")

    adata = sc.read_h5ad(
        ROOT / "data" / f"{stage}.h5ad"
    )

    xyz = np.asarray(
        adata.obsm["spatial_3D"],
        dtype=np.float64
    )

    labels = adata.obs["celltype"].astype(str).to_numpy()
    celltypes = sorted(set(labels))
    celltype_sets[stage] = set(celltypes)

    print("Cells:", adata.n_obs)
    print("Annotated cell types:", len(celltypes))

    for ct in celltypes:
        mask = labels == ct
        points = xyz[mask]

        n = len(points)
        centroid = points.mean(axis=0)
        centered = points - centroid

        rg = np.sqrt(
            np.mean(
                np.sum(centered ** 2, axis=1)
            )
        )

        if n >= 3:
            covariance = np.cov(
                centered,
                rowvar=False
            )

            eigenvalues = np.linalg.eigvalsh(
                covariance
            )[::-1]

            axis_std = np.sqrt(
                np.maximum(eigenvalues, 0)
            )
        else:
            axis_std = np.full(3, np.nan)

        records.append({
            "stage": stage,
            "celltype": ct,
            "n_cells": n,
            "fraction": n / adata.n_obs,
            "centroid_x": centroid[0],
            "centroid_y": centroid[1],
            "centroid_z": centroid[2],
            "radius_gyration": rg,
            "axis1_std": axis_std[0],
            "axis2_std": axis_std[1],
            "axis3_std": axis_std[2],
        })

summary = pd.DataFrame(records)

summary.to_csv(
    OUT / "celltype_geometry_summary.csv",
    index=False
)

shared = sorted(
    celltype_sets["E7.25"]
    & celltype_sets["E8.0"]
)

only_725 = sorted(
    celltype_sets["E7.25"]
    - celltype_sets["E8.0"]
)

only_800 = sorted(
    celltype_sets["E8.0"]
    - celltype_sets["E7.25"]
)

print("\n" + "=" * 60)
print("CELL TYPE COMPARISON")
print("=" * 60)

print("\nShared cell types:", len(shared))
print(shared)

print("\nOnly E7.25:", only_725)
print("\nOnly E8.0:", only_800)

print("\n" + "=" * 60)
print("SHARED CELL-TYPE GEOMETRY")
print("=" * 60)

for ct in shared:
    early = summary[
        (summary.stage == "E7.25")
        & (summary.celltype == ct)
    ].iloc[0]

    late = summary[
        (summary.stage == "E8.0")
        & (summary.celltype == ct)
    ].iloc[0]

    ratio = (
        late.radius_gyration
        / early.radius_gyration
        if early.radius_gyration > 0
        else np.nan
    )

    print(f"\n{ct}")
    print(
        f"  Cells: {early.n_cells}"
        f" -> {late.n_cells}"
    )
    print(
        f"  Radius: {early.radius_gyration:.2f}"
        f" -> {late.radius_gyration:.2f}"
    )
    print(f"  Radius ratio: {ratio:.3f}")

print("\nSaved:",
      OUT / "celltype_geometry_summary.csv")
