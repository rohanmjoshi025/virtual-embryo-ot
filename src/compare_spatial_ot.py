
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt

from scipy import sparse
from scipy.spatial import cKDTree
from moscot.problems.time import TemporalProblem


# ============================================================
# 1. SETTINGS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = RESULTS / "spatial_ot_comparison"
OUT.mkdir(parents=True, exist_ok=True)

SOURCE_TIME = 6.75
TARGET_TIME = 7.25

EPSILON = 0.05
MAX_ITERATIONS = 2000

RANDOM_SEED = 42
N_ARROWS = 100

# Spatial feature weights
MODELS = {
    "RNA_only": 0.0,
    "Spatial_low": 0.25,
    "Spatial_medium": 0.50,
    "Spatial_high": 1.0,
}


# ============================================================
# 2. LOAD DATA
# ============================================================

print("Loading pilot dataset...")

adata = sc.read_h5ad(
    RESULTS / "embryo_pilot.h5ad"
)

assert "X_pca" in adata.obsm
assert "spatial_3D" in adata.obsm
assert "celltype" in adata.obs
assert "time" in adata.obs

times = adata.obs["time"].astype(float).to_numpy()

source_mask = np.isclose(times, SOURCE_TIME)
target_mask = np.isclose(times, TARGET_TIME)

assert source_mask.sum() > 0
assert target_mask.sum() > 0

# Restrict to one transition so all scaling and
# comparisons use exactly the same cell populations.
pair = adata[source_mask | target_mask].copy()

pair_times = pair.obs["time"].astype(float).to_numpy()

source_mask = np.isclose(pair_times, SOURCE_TIME)
target_mask = np.isclose(pair_times, TARGET_TIME)

rna = np.asarray(pair.obsm["X_pca"], dtype=float)
xyz = np.asarray(pair.obsm["spatial_3D"], dtype=float)

assert np.isfinite(rna).all()
assert np.isfinite(xyz).all()
assert xyz.shape[1] == 3

print("Source cells:", source_mask.sum())
print("Target cells:", target_mask.sum())
print("RNA PCs:", rna.shape[1])


# ============================================================
# 3. NORMALIZE RNA AND SPATIAL FEATURES
# ============================================================

# Important:
# Use shared scaling across both developmental stages.
# Do NOT center or scale each stage independently.
#
# We normalize the two feature groups to have
# comparable average squared distances.
#
# This preserves the relative contribution of
# individual RNA PCs within the RNA feature group.

def normalize_feature_group(X):

    X = np.asarray(X, dtype=float)

    center = X.mean(axis=0)
    centered = X - center

    # Root mean squared Euclidean distance to
    # the pooled centroid.
    scale = np.sqrt(
        np.mean(np.sum(centered ** 2, axis=1))
    )

    if scale <= 0:
        raise ValueError("Feature group has zero variance")

    return centered / scale


rna_scaled = normalize_feature_group(rna)
xyz_scaled = normalize_feature_group(xyz)

# Keep original coordinates for displacement analysis.
source_xyz = xyz[source_mask]
target_xyz = xyz[target_mask]

source_types = (
    pair.obs.loc[source_mask, "celltype"]
    .astype(str)
    .to_numpy()
)

target_types = (
    pair.obs.loc[target_mask, "celltype"]
    .astype(str)
    .to_numpy()
)

# Create consistent cell-type ordering
all_types = sorted(
    set(source_types) | set(target_types)
)


# ============================================================
# 4. HELPER: RUN OT
# ============================================================

def run_ot(spatial_weight):

    # Combine normalized RNA PCs and spatial coordinates.
    # RNA feature group always has weight 1.
    combined = np.concatenate(
        [
            rna_scaled,
            spatial_weight * xyz_scaled,
        ],
        axis=1,
    )

    # Copy data to avoid overwriting other model inputs.
    model_data = pair.copy()

    model_data.obsm["X_combined"] = combined

    problem = TemporalProblem(model_data)

    problem = problem.prepare(
        time_key="time",
        joint_attr="X_combined",
        policy="sequential",
        a=False,
        b=False,
    )

    problem = problem.solve(
        epsilon=EPSILON,
        max_iterations=MAX_ITERATIONS,
    )

    sub = problem.problems[
        (SOURCE_TIME, TARGET_TIME)
    ]

    if not sub.solution.converged:
        print("WARNING: OT solver did not converge")

    P = sub.solution.transport_matrix

    if sparse.issparse(P):
        P = P.toarray()
    else:
        P = np.asarray(P, dtype=float)

    if P.shape != (
        len(source_xyz),
        len(target_xyz),
    ):
        raise ValueError(
            f"Unexpected transport shape: {P.shape}"
        )

    if not np.isfinite(P).all() or np.any(P < 0):
        raise ValueError("Invalid transport matrix")

    return P, bool(sub.solution.converged)


# ============================================================
# 5. HELPER: TRANSPORT DIAGNOSTICS
# ============================================================

def calculate_diagnostics(P):

    row_mass = P.sum(axis=1)

    valid = row_mass > 1e-12

    q = np.divide(
        P,
        row_mass[:, None],
        out=np.zeros_like(P, dtype=float),
        where=row_mass[:, None] > 1e-12,
    )

    # Barycentric target positions
    predicted_xyz = q @ target_xyz

    predicted_xyz[~valid] = np.nan

    # Displacement
    displacement = predicted_xyz - source_xyz

    magnitude = np.linalg.norm(
        displacement,
        axis=1,
    )

    # Entropy
    log_q = np.zeros_like(q)

    np.log(
        q,
        out=log_q,
        where=q > 0,
    )

    entropy = -np.sum(
        q * log_q,
        axis=1,
    )

    entropy[~valid] = np.nan

    effective_targets = np.exp(entropy)

    max_weight = q.max(axis=1)
    max_weight[~valid] = np.nan

    # Target spatial spread
    target_norm_sq = np.sum(
        target_xyz ** 2,
        axis=1,
    )

    expected_norm_sq = q @ target_norm_sq

    barycenter_norm_sq = np.sum(
        np.nan_to_num(predicted_xyz) ** 2,
        axis=1,
    )

    spatial_variance = (
        expected_norm_sq - barycenter_norm_sq
    )

    spatial_spread = np.sqrt(
        np.maximum(spatial_variance, 0)
    )

    spatial_spread[~valid] = np.nan

    return {
        "q": q,
        "valid": valid,
        "predicted_xyz": predicted_xyz,
        "displacement": displacement,
        "magnitude": magnitude,
        "entropy": entropy,
        "effective_targets": effective_targets,
        "max_weight": max_weight,
        "spatial_spread": spatial_spread,
    }


# ============================================================
# 6. HELPER: LOCAL DIRECTIONAL COHERENCE
# ============================================================

def local_coherence(displacement, k=10):

    # Compare each source cell's displacement direction
    # with its spatial neighbors.
    #
    # +1: strongly aligned directions
    #  0: little directional alignment
    # -1: opposing directions

    norms = np.linalg.norm(
        displacement,
        axis=1,
    )

    valid = (
        np.isfinite(norms)
        & (norms > 1e-10)
    )

    directions = np.zeros_like(displacement)

    directions[valid] = (
        displacement[valid] / norms[valid, None]
    )

    tree = cKDTree(source_xyz)

    k_query = min(k + 1, len(source_xyz))

    _, neighbors = tree.query(
        source_xyz,
        k=k_query,
    )

    neighbors = np.atleast_2d(neighbors)

    # Exclude each cell itself
    neighbors = neighbors[:, 1:]

    similarities = np.sum(
        directions[:, None, :]
        * directions[neighbors],
        axis=2,
    )

    valid_pairs = (
        valid[:, None]
        & valid[neighbors]
    )

    if not np.any(valid_pairs):
        return np.nan

    return float(
        similarities[valid_pairs].mean()
    )


# ============================================================
# 7. HELPER: CELL-TYPE TRANSITION MATRIX
# ============================================================

def celltype_transition(P):

    matrix = pd.DataFrame(
        0.0,
        index=all_types,
        columns=all_types,
    )

    source_codes = pd.Categorical(
        source_types,
        categories=all_types,
    ).codes

    target_codes = pd.Categorical(
        target_types,
        categories=all_types,
    ).codes

    mass = np.zeros(
        (len(all_types), len(all_types)),
        dtype=float,
    )

    # Aggregate transport mass by source and target type
    np.add.at(
        mass,
        (
            np.repeat(source_codes, len(target_codes)),
            np.tile(target_codes, len(source_codes)),
        ),
        P.ravel(),
    )

    # Normalize rows to percentages
    row_sums = mass.sum(axis=1, keepdims=True)

    percentages = np.divide(
        mass,
        row_sums,
        out=np.zeros_like(mass),
        where=row_sums > 0,
    ) * 100

    matrix.iloc[:, :] = percentages

    return matrix


# ============================================================
# 8. RUN ALL FOUR MODELS
# ============================================================

summary = []
results = {}
transitions = {}

for model_name, spatial_weight in MODELS.items():

    print("\n" + "=" * 60)
    print("Running:", model_name)
    print("Spatial weight:", spatial_weight)
    print("=" * 60)

    P, converged = run_ot(spatial_weight)

    diagnostics = calculate_diagnostics(P)

    coherence = local_coherence(
        diagnostics["displacement"]
    )

    transition = celltype_transition(P)

    transitions[model_name] = transition
    results[model_name] = diagnostics

    summary.append({
        "model": model_name,
        "spatial_weight": spatial_weight,
        "converged": converged,
        "mean_displacement": np.nanmean(
            diagnostics["magnitude"]
        ),
        "median_displacement": np.nanmedian(
            diagnostics["magnitude"]
        ),
        "mean_effective_targets": np.nanmean(
            diagnostics["effective_targets"]
        ),
        "mean_max_weight": np.nanmean(
            diagnostics["max_weight"]
        ),
        "mean_spatial_spread": np.nanmean(
            diagnostics["spatial_spread"]
        ),
        "median_spatial_spread": np.nanmedian(
            diagnostics["spatial_spread"]
        ),
        "local_coherence": coherence,
    })

    # Save transition percentages
    transition.to_csv(
        OUT / f"transitions_{model_name}.csv"
    )

    # Save per-cell diagnostics
    cell_df = pd.DataFrame({
        "celltype": source_types,
        "source_x": source_xyz[:, 0],
        "source_y": source_xyz[:, 1],
        "source_z": source_xyz[:, 2],
        "dx": diagnostics["displacement"][:, 0],
        "dy": diagnostics["displacement"][:, 1],
        "dz": diagnostics["displacement"][:, 2],
        "displacement_magnitude": diagnostics["magnitude"],
        "effective_targets": diagnostics["effective_targets"],
        "max_target_weight": diagnostics["max_weight"],
        "target_spatial_spread": diagnostics["spatial_spread"],
    })

    cell_df.to_csv(
        OUT / f"cells_{model_name}.csv",
        index=False,
    )


# ============================================================
# 9. SAVE MODEL COMPARISON TABLE
# ============================================================

summary_df = pd.DataFrame(summary)

summary_df.to_csv(
    OUT / "model_comparison.csv",
    index=False,
)

print("\nMODEL COMPARISON")
print(
    summary_df.round(3).to_string(index=False)
)


# ============================================================
# 10. PLOT COMPARATIVE DISPLACEMENT FIELDS
# ============================================================

rng = np.random.default_rng(RANDOM_SEED)

indices = rng.choice(
    len(source_xyz),
    size=min(N_ARROWS, len(source_xyz)),
    replace=False,
)

fig, axes = plt.subplots(
    2, 2,
    figsize=(15, 13),
    sharex=True,
    sharey=True,
)

# Consistent color range across models
all_magnitudes = np.concatenate([
    r["magnitude"][np.isfinite(r["magnitude"])]
    for r in results.values()
])

vmax = np.percentile(all_magnitudes, 95)

for ax, (model_name, diagnostics) in zip(
    axes.flat,
    results.items(),
):

    d = diagnostics["displacement"]
    m = diagnostics["magnitude"]

    ax.scatter(
        source_xyz[:, 0],
        source_xyz[:, 1],
        s=3,
        color="lightgray",
        alpha=0.25,
    )

    qplot = ax.quiver(
        source_xyz[indices, 0],
        source_xyz[indices, 1],
        d[indices, 0],
        d[indices, 1],
        m[indices],
        cmap="viridis",
        clim=(0, vmax),
        angles="xy",
        scale_units="xy",
        scale=1,
    )

    ax.set_title(model_name.replace("_", " "))
    ax.set_xlabel("Spatial X")
    ax.set_ylabel("Spatial Y")
    ax.set_aspect("equal", adjustable="box")

fig.colorbar(
    qplot,
    ax=axes.ravel().tolist(),
    label="3D displacement magnitude",
    shrink=0.75,
)

fig.suptitle(
    "E6.75 to E7.25: RNA versus Spatial OT",
    fontsize=16,
)

fig.subplots_adjust(
    left=0.07,
    right=0.88,
    bottom=0.06,
    top=0.92,
    wspace=0.15,
    hspace=0.20,
)

fig.savefig(
    OUT / "displacement_comparison.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# 11. PLOT DIAGNOSTIC COMPARISONS
# ============================================================

metrics = [
    ("mean_displacement", "Mean displacement"),
    ("mean_effective_targets", "Effective targets"),
    ("mean_spatial_spread", "Target spatial spread"),
    ("local_coherence", "Local directional coherence"),
]

fig, axes = plt.subplots(
    2, 2,
    figsize=(13, 9),
)

for ax, (column, title) in zip(
    axes.flat,
    metrics,
):

    ax.plot(
        summary_df["spatial_weight"],
        summary_df[column],
        marker="o",
        linewidth=2,
    )

    ax.set_xlabel("Spatial feature weight")
    ax.set_ylabel(title)
    ax.set_title(title)
    ax.grid(alpha=0.25)

plt.tight_layout()

plt.savefig(
    OUT / "diagnostic_comparison.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# 12. COMPARE CELL-TYPE TRANSITIONS
# ============================================================

# Compare every model's transition matrix
# against the RNA-only baseline.

baseline = transitions["RNA_only"]

transition_differences = []

for model_name, transition in transitions.items():

    # Only compare rows representing source cell types
    source_present = [
        t for t in all_types
        if np.any(source_types == t)
    ]

    difference = (
        transition.loc[source_present]
        - baseline.loc[source_present]
    )

    mean_absolute_change = (
        difference.abs().to_numpy().mean()
    )

    transition_differences.append({
        "model": model_name,
        "mean_absolute_transition_change_pp":
            mean_absolute_change,
    })

difference_df = pd.DataFrame(
    transition_differences
)

difference_df.to_csv(
    OUT / "transition_differences.csv",
    index=False,
)

print("\nCELL-TYPE TRANSITION CHANGES")
print(
    difference_df.round(3).to_string(index=False)
)


# ============================================================
# 13. FINISHED
# ============================================================

print("\nAnalysis complete.")
print("All results saved to:")
print(OUT)
