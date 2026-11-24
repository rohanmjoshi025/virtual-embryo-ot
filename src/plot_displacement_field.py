from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt

from scipy import sparse
from moscot.problems.time import TemporalProblem


# ============================================================
# 1. FILE PATHS AND SETTINGS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

DATA_FILE = RESULTS / "embryo_pilot.h5ad"

SOURCE_TIME = 6.75
TARGET_TIME = 7.25

EPSILON = 0.05
MAX_ITERATIONS = 2000

N_ARROWS = 150
RANDOM_SEED = 42

LABEL = f"E{SOURCE_TIME}_to_E{TARGET_TIME}"


# ============================================================
# 2. LOAD DATA
# ============================================================

print("\nLoading embryo dataset...")

adata = sc.read_h5ad(DATA_FILE)

assert "X_pca" in adata.obsm, "Missing RNA PCA coordinates"
assert "spatial_3D" in adata.obsm, "Missing 3D coordinates"
assert "time" in adata.obs, "Missing developmental time"

print("Total cells:", adata.n_obs)
print("Total genes:", adata.n_vars)
print("PCA shape:", adata.obsm["X_pca"].shape)
print("Spatial shape:", adata.obsm["spatial_3D"].shape)


# ============================================================
# 3. RUN RNA-ONLY OPTIMAL TRANSPORT
# ============================================================

print("\nPreparing RNA-only temporal OT...")

problem = TemporalProblem(adata)

problem = problem.prepare(
    time_key="time",
    joint_attr="X_pca",
    policy="sequential",
    a=False,
    b=False,
)

print("Solving optimal transport...")

problem = problem.solve(
    epsilon=EPSILON,
    max_iterations=MAX_ITERATIONS,
)

subproblem = problem.problems[
    (SOURCE_TIME, TARGET_TIME)
]

print(
    "Converged:",
    subproblem.solution.converged
)


# ============================================================
# 4. EXTRACT TRANSPORT MATRIX AND SPATIAL COORDINATES
# ============================================================

P = subproblem.solution.transport_matrix

if sparse.issparse(P):
    P = P.toarray()
else:
    P = np.asarray(P, dtype=float)

source_xyz = np.asarray(
    subproblem.adata_src.obsm["spatial_3D"],
    dtype=float,
)

target_xyz = np.asarray(
    subproblem.adata_tgt.obsm["spatial_3D"],
    dtype=float,
)

source_types = (
    subproblem.adata_src.obs["celltype"]
    .astype(str)
    .to_numpy()
)

print("\nTransport matrix:", P.shape)
print("Source coordinates:", source_xyz.shape)
print("Target coordinates:", target_xyz.shape)

assert P.shape == (
    source_xyz.shape[0],
    target_xyz.shape[0],
)

assert source_xyz.shape[1] == 3
assert target_xyz.shape[1] == 3

assert np.all(np.isfinite(P))
assert np.all(P >= 0)
assert np.all(np.isfinite(source_xyz))
assert np.all(np.isfinite(target_xyz))


# ============================================================
# 5. NORMALIZE TRANSPORT WEIGHTS
# ============================================================

# Each row describes how one source cell distributes its
# transport mass across all target cells.

row_mass = P.sum(axis=1)

valid = row_mass > 0

q = np.divide(
    P,
    row_mass[:, None],
    out=np.zeros_like(P, dtype=float),
    where=row_mass[:, None] > 0,
)

print("\nValid source cells:", valid.sum())


# ============================================================
# 6. CALCULATE BARYCENTRIC DISPLACEMENT
# ============================================================

# Weighted average target position for each source cell.

predicted_xyz = np.full(
    source_xyz.shape,
    np.nan,
    dtype=float,
)

predicted_xyz[valid] = (
    q[valid] @ target_xyz
)

# Displacement vector from source to weighted target position.

displacement = predicted_xyz - source_xyz

dx = displacement[:, 0]
dy = displacement[:, 1]
dz = displacement[:, 2]

magnitude = np.linalg.norm(
    displacement,
    axis=1,
)


# ============================================================
# 7. TRANSPORT ENTROPY
# ============================================================

# H_i = -sum_j q_ij log(q_ij)
#
# Low entropy: mass concentrated on fewer targets
# High entropy: mass distributed over many targets

entropy = np.full(
    len(source_xyz),
    np.nan,
    dtype=float,
)

positive_q = q > 0

log_q = np.zeros_like(q)

np.log(
    q,
    out=log_q,
    where=positive_q,
)

entropy[valid] = -np.sum(
    q[valid] * log_q[valid],
    axis=1,
)

# Effective number of targets:
# N_eff = exp(H)

effective_targets = np.exp(entropy)

# Largest normalized target weight per source cell

max_weight = np.full(
    len(source_xyz),
    np.nan,
    dtype=float,
)

max_weight[valid] = q[valid].max(axis=1)


# ============================================================
# 8. SPATIAL SPREAD OF TARGET ASSIGNMENTS
# ============================================================

# S_i = sqrt(sum_j q_ij * ||x_j - predicted_x_i||^2)
#
# This measures how spatially dispersed the target
# assignments are around their barycentric mean.

target_norm_sq = np.sum(
    target_xyz ** 2,
    axis=1,
)

expected_target_norm_sq = (
    q @ target_norm_sq
)

barycenter_norm_sq = np.sum(
    np.nan_to_num(predicted_xyz) ** 2,
    axis=1,
)

spatial_variance = (
    expected_target_norm_sq
    - barycenter_norm_sq
)

spatial_spread = np.full(
    len(source_xyz),
    np.nan,
    dtype=float,
)

spatial_spread[valid] = np.sqrt(
    np.maximum(
        spatial_variance[valid],
        0,
    )
)


# ============================================================
# 9. SAVE RESULTS TO CSV
# ============================================================

df = pd.DataFrame({
    "celltype": source_types,

    "source_x": source_xyz[:, 0],
    "source_y": source_xyz[:, 1],
    "source_z": source_xyz[:, 2],

    "predicted_x": predicted_xyz[:, 0],
    "predicted_y": predicted_xyz[:, 1],
    "predicted_z": predicted_xyz[:, 2],

    "dx": dx,
    "dy": dy,
    "dz": dz,

    "displacement_magnitude": magnitude,

    "transport_entropy": entropy,
    "effective_targets": effective_targets,
    "max_target_weight": max_weight,

    "target_spatial_spread": spatial_spread,
})

csv_path = RESULTS / f"displacement_diagnostics_{LABEL}.csv"

df.to_csv(csv_path, index=False)

print("\nSaved:", csv_path)


# ============================================================
# 10. PRINT DIAGNOSTIC STATISTICS
# ============================================================

print("\n" + "=" * 55)
print(f"DISPLACEMENT DIAGNOSTICS: {LABEL}")
print("=" * 55)

print("\nDisplacement magnitude:")
print("Mean:", np.nanmean(magnitude))
print("Median:", np.nanmedian(magnitude))
print("Maximum:", np.nanmax(magnitude))

print("\nTransport entropy:")
print("Mean:", np.nanmean(entropy))
print("Median:", np.nanmedian(entropy))

print("\nEffective number of targets:")
print("Mean:", np.nanmean(effective_targets))
print("Median:", np.nanmedian(effective_targets))

print("\nMaximum target weight:")
print("Mean:", np.nanmean(max_weight))
print("Median:", np.nanmedian(max_weight))

print("\nTarget spatial spread:")
print("Mean:", np.nanmean(spatial_spread))
print("Median:", np.nanmedian(spatial_spread))


# ============================================================
# 11. SAMPLE CELLS FOR VISUALIZATION
# ============================================================

rng = np.random.default_rng(RANDOM_SEED)

indices = np.flatnonzero(valid)

if len(indices) > N_ARROWS:
    indices = rng.choice(
        indices,
        size=N_ARROWS,
        replace=False,
    )

x = source_xyz[indices]
d = displacement[indices]
m = magnitude[indices]


# ============================================================
# 12. 3D DISPLACEMENT FIELD
# ============================================================

fig = plt.figure(figsize=(12, 10))

ax = fig.add_subplot(
    111,
    projection="3d",
)

# Background source cells

ax.scatter(
    source_xyz[:, 0],
    source_xyz[:, 1],
    source_xyz[:, 2],
    color="lightgray",
    alpha=0.15,
    s=3,
)

# Displacement vectors

ax.quiver(
    x[:, 0],
    x[:, 1],
    x[:, 2],

    d[:, 0],
    d[:, 1],
    d[:, 2],

    length=1,
    normalize=False,

    color="royalblue",
    linewidth=0.8,
    arrow_length_ratio=0.08,
)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

ax.set_title(
    f"RNA-based OT displacement: "
    f"E{SOURCE_TIME} to E{TARGET_TIME}"
)

ax.view_init(
    elev=20,
    azim=-60,
)

plt.tight_layout()

path_3d = RESULTS / f"displacement_3D_{LABEL}.png"

plt.savefig(
    path_3d,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)

print("\nSaved:", path_3d)


# ============================================================
# 13. 2D PROJECTED DISPLACEMENT FIELD
# ============================================================

fig, ax = plt.subplots(
    figsize=(11, 9)
)

ax.scatter(
    source_xyz[:, 0],
    source_xyz[:, 1],
    color="lightgray",
    alpha=0.2,
    s=4,
)

quiver = ax.quiver(
    x[:, 0],
    x[:, 1],

    d[:, 0],
    d[:, 1],

    m,

    angles="xy",
    scale_units="xy",
    scale=1,

    cmap="viridis",
)

fig.colorbar(
    quiver,
    ax=ax,
    label="3D displacement magnitude",
)

ax.set_xlabel("Spatial X")
ax.set_ylabel("Spatial Y")

ax.set_title(
    f"Projected displacement: "
    f"E{SOURCE_TIME} to E{TARGET_TIME}"
)

ax.set_aspect(
    "equal",
    adjustable="box",
)

plt.tight_layout()

path_2d = RESULTS / f"displacement_2D_{LABEL}.png"

plt.savefig(
    path_2d,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)

print("Saved:", path_2d)


# ============================================================
# 14. EFFECTIVE TARGET COUNT HISTOGRAM
# ============================================================

fig, ax = plt.subplots(figsize=(9, 6))

ax.hist(
    effective_targets[valid],
    bins=40,
    edgecolor="black",
    alpha=0.8,
)

ax.set_xlabel("Effective number of target cells")
ax.set_ylabel("Number of source cells")

ax.set_title(
    "Distribution of Effective Target Counts"
)

plt.tight_layout()

path_entropy = RESULTS / f"effective_targets_{LABEL}.png"

plt.savefig(
    path_entropy,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)

print("Saved:", path_entropy)


# ============================================================
# 15. SPATIAL SPREAD HISTOGRAM
# ============================================================

fig, ax = plt.subplots(figsize=(9, 6))

ax.hist(
    spatial_spread[valid],
    bins=40,
    edgecolor="black",
    alpha=0.8,
)

ax.set_xlabel("Target spatial spread")
ax.set_ylabel("Number of source cells")

ax.set_title(
    "Spatial Spread of OT Target Assignments"
)

plt.tight_layout()

path_spread = RESULTS / f"target_spatial_spread_{LABEL}.png"

plt.savefig(
    path_spread,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)

print("Saved:", path_spread)


# ============================================================
# 16. FINISHED
# ============================================================

print("\n" + "=" * 55)
print("ANALYSIS COMPLETE")
print("=" * 55)

print("\nAll outputs saved in:")
print(RESULTS)


