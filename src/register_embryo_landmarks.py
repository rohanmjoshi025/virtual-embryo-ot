from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent

OUT = ROOT / "results" / "celltype_geometry"
OUT.mkdir(parents=True, exist_ok=True)

ANCHORS = [
    "Allantois",
    "Caudal Epiblast",
    "LPM",
    "Primitive Streak",
    "SOM",
]

print("Loading E7.25 and E8.0...")

early = sc.read_h5ad(ROOT / "data/E7.25.h5ad")
late = sc.read_h5ad(ROOT / "data/E8.0.h5ad")


def normalized_geometry(adata):
    xyz = np.asarray(
        adata.obsm["spatial_3D"],
        dtype=np.float64
    )

    center = xyz.mean(axis=0)
    rg = np.sqrt(np.mean(
        np.sum((xyz - center) ** 2, axis=1)
    ))

    # Normalize overall size and position.
    xyz_norm = (xyz - center) / rg

    labels = adata.obs["celltype"].astype(str).to_numpy()

    centroids = []
    counts = []

    for ct in ANCHORS:
        mask = labels == ct

        if not mask.any():
            raise ValueError(f"Missing anchor: {ct}")

        counts.append(int(mask.sum()))
        centroids.append(xyz_norm[mask].mean(axis=0))

    return xyz_norm, np.asarray(centroids), counts


xyz_a, A, counts_a = normalized_geometry(early)
xyz_b, B, counts_b = normalized_geometry(late)


def fit_rigid(A, B):
    """
    Find the rotation and translation mapping
    A onto B, without reflection or extra scaling.

    Coordinates have already been normalized
    by each embryo's global radius of gyration.
    """

    mean_a = A.mean(axis=0)
    mean_b = B.mean(axis=0)

    Ac = A - mean_a
    Bc = B - mean_b

    H = Ac.T @ Bc

    U, _, Vt = np.linalg.svd(H)

    D = np.eye(3)
    D[-1, -1] = np.linalg.det(U @ Vt)

    R = U @ D @ Vt

    t = mean_b - mean_a @ R

    return R, t


# Fit using all candidate landmarks.
R, t = fit_rigid(A, B)

A_aligned = A @ R + t

errors = np.linalg.norm(
    A_aligned - B,
    axis=1
)

print("\n" + "=" * 55)
print("LANDMARK ALIGNMENT")
print("=" * 55)

for i, ct in enumerate(ANCHORS):
    print(
        f"{ct:20s} "
        f"E7.25={counts_a[i]:5d} "
        f"E8.0={counts_b[i]:5d} "
        f"error={errors[i]:.4f}"
    )

print("\nMean normalized landmark error:",
      round(float(errors.mean()), 4))


# Compare landmark relationships before alignment.
# Pairwise distances do not depend on rotation.
dist_a = pdist(A)
dist_b = pdist(B)

correlation = np.corrcoef(dist_a, dist_b)[0, 1]

print("\nPairwise landmark distance correlation:",
      round(float(correlation), 4))


# Leave-one-landmark-out validation.
# This tests whether an alignment estimated from
# four landmarks predicts the fifth landmark.
print("\n" + "=" * 55)
print("LEAVE-ONE-LANDMARK-OUT VALIDATION")
print("=" * 55)

records = []

for i, ct in enumerate(ANCHORS):
    keep = np.arange(len(ANCHORS)) != i

    R_test, t_test = fit_rigid(
        A[keep],
        B[keep]
    )

    predicted = A[i] @ R_test + t_test

    error = np.linalg.norm(predicted - B[i])

    records.append({
        "celltype": ct,
        "held_out_error": error
    })

    print(f"{ct:20s} error={error:.4f}")

results = pd.DataFrame(records)

results.to_csv(
    OUT / "landmark_registration_errors.csv",
    index=False
)

print("\nMean held-out error:",
      round(float(results.held_out_error.mean()), 4))


# Visualize the registered point clouds.
rng = np.random.default_rng(42)

idx_a = rng.choice(
    len(xyz_a),
    min(3500, len(xyz_a)),
    replace=False
)

idx_b = rng.choice(
    len(xyz_b),
    min(3500, len(xyz_b)),
    replace=False
)

aligned_xyz_a = xyz_a @ R + t

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

ax.scatter(
    *aligned_xyz_a[idx_a].T,
    s=0.6,
    alpha=0.12,
    color="royalblue",
    label="E7.25 aligned"
)

ax.scatter(
    *xyz_b[idx_b].T,
    s=0.6,
    alpha=0.12,
    color="darkorange",
    label="E8.0"
)

for i, ct in enumerate(ANCHORS):
    ax.scatter(
        *A_aligned[i],
        color="blue",
        s=35
    )
    ax.scatter(
        *B[i],
        color="red",
        s=35,
        marker="x"
    )

    ax.text(
        *B[i],
        ct,
        fontsize=8
    )

ax.set_title(
    "E7.25 vs E8.0\n"
    "Landmark-based registration (normalized)"
)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.legend()

plt.tight_layout()

figure_path = OUT / "landmark_registration.png"

plt.savefig(
    figure_path,
    dpi=200,
    bbox_inches="tight"
)

print("\nSaved:", figure_path)
print("Registration diagnostic complete.")

plt.show()
