from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "submission"

ALPHA = 2 / 3
SEED = 42
V2_SCALE = 0.75

ANCHORS = [
    "Allantois",
    "Caudal Epiblast",
    "LPM",
    "Primitive Streak",
    "SOM",
]

# Reduce the influence of the least reliable
# landmark identified by cross-validation.
ANCHOR_WEIGHTS = np.array(
    [0.2, 1.0, 1.0, 1.0, 1.0]
)

print("Loading data...")

early = sc.read_h5ad(ROOT / "data/E7.25.h5ad")
late = sc.read_h5ad(ROOT / "data/E8.0.h5ad")
v2 = sc.read_h5ad(OUT / "e775_v2_geometry.h5ad")

xyz_a = np.asarray(
    early.obsm["spatial_3D"], dtype=np.float64
)
xyz_b = np.asarray(
    late.obsm["spatial_3D"], dtype=np.float64
)

labels_a = early.obs["celltype"].astype(str).to_numpy()
labels_b = late.obs["celltype"].astype(str).to_numpy()

def radius(points):
    center = points.mean(axis=0)
    return np.sqrt(np.mean(
        np.sum((points - center) ** 2, axis=1)
    ))

center_a = xyz_a.mean(axis=0)
center_b = xyz_b.mean(axis=0)

rg_a = radius(xyz_a)
rg_b = radius(xyz_b)

# Recover the exact E8.0 donor cell indices used
# by the original V2 generator.
rng = np.random.default_rng(SEED)

target_idx = rng.choice(
    late.n_obs,
    size=v2.n_obs,
    replace=False
)

target_xyz = xyz_b[target_idx]
target_labels = labels_b[target_idx]

# Verify these indices really reproduce V2.
v2_center = target_xyz.mean(axis=0)
reconstructed_v2 = (
    v2_center + V2_SCALE * (target_xyz - v2_center)
)

if not np.allclose(
    np.asarray(v2.obsm["spatial_3D"]),
    reconstructed_v2,
    atol=0.01
):
    raise RuntimeError(
        "V2 donor indices could not be recovered. "
        "Check the original V2 seed and sampling code."
    )

print("V2 donor positions verified.")

def points_by_type(xyz, labels):
    return {
        ct: xyz[labels == ct]
        for ct in np.unique(labels)
    }

types_a = points_by_type(xyz_a, labels_a)
types_b = points_by_type(xyz_b, labels_b)

# Normalize anchor centers by each embryo's
# global radius of gyration.
A = np.stack([
    (types_a[ct].mean(axis=0) - center_a) / rg_a
    for ct in ANCHORS
])

B = np.stack([
    (types_b[ct].mean(axis=0) - center_b) / rg_b
    for ct in ANCHORS
])

def weighted_registration(A, B, weights):
    w = weights / weights.sum()

    ma = np.sum(A * w[:, None], axis=0)
    mb = np.sum(B * w[:, None], axis=0)

    Ac = A - ma
    Bc = B - mb

    H = (Ac * w[:, None]).T @ Bc

    U, _, Vt = np.linalg.svd(H)

    # Disallow reflection.
    D = np.eye(3)
    D[-1, -1] = np.sign(np.linalg.det(U @ Vt))

    R = U @ D @ Vt
    t = mb - ma @ R

    return R, t

R, t = weighted_registration(
    A, B, ANCHOR_WEIGHTS
)

print("Registration estimated.")

# ---------------------------------------------
# Candidate 1: global log-scale interpolation
# ---------------------------------------------

rg_pred = rg_a ** (1 - ALPHA) * rg_b ** ALPHA
global_scale = rg_pred / rg_b

xyz_global = (
    center_b
    + global_scale * (target_xyz - center_b)
)

print("Global scale:", round(global_scale, 4))
print("Target global Rg:", round(rg_pred, 2))

# ---------------------------------------------
# Candidate 2: tissue-aware geometry
# ---------------------------------------------

xyz_tissue = xyz_global.copy()

shared = sorted(set(types_a) & set(types_b))

for ct in shared:

    # Avoid very small populations and unknown labels.
    if ct == "Unknown":
        continue

    pa = types_a[ct]
    pb = types_b[ct]

    if len(pa) < 100 or len(pb) < 100:
        continue

    mask = target_labels == ct

    if not np.any(mask):
        continue

    ca = pa.mean(axis=0)
    cb = pb.mean(axis=0)

    # Map E7.25 tissue centroid into the
    # E8.0 orientation and coordinate frame.
    ca_normalized = (ca - center_a) / rg_a

    ca_aligned = (
        center_b
        + rg_a * (ca_normalized @ R + t)
    )

    # Intermediate tissue centroid.
    c_pred = (
        (1 - ALPHA) * ca_aligned
        + ALPHA * cb
    )

    ra = radius(pa)
    rb = radius(pb)

    # Geometric interpolation of tissue spread.
    r_pred = ra ** (1 - ALPHA) * rb ** ALPHA
    tissue_scale = r_pred / max(rb, 1e-8)

    # Preserve E8.0 within-tissue organization
    # while updating centroid and spread.
    candidate = (
        c_pred
        + tissue_scale * (target_xyz[mask] - cb)
    )

    # Conservative blend to reduce artifacts
    # from imperfect stage registration.
    strength = 0.25 if ct == "Allantois" else 0.55

    xyz_tissue[mask] = (
        (1 - strength) * xyz_global[mask]
        + strength * candidate
    )

    print(
        f"{ct:20s} "
        f"cells={mask.sum():5d} "
        f"scale={tissue_scale:.3f}"
    )

# ---------------------------------------------
# Save candidates with V2 expression unchanged
# ---------------------------------------------

def save_candidate(name, xyz):
    pred = v2.copy()
    pred.obs = pd.DataFrame(index=pred.obs_names.copy())
    pred.obsm["spatial_3D"] = xyz.astype(np.float32)

    assert pred.n_vars == 498
    assert pred.obsm["spatial_3D"].shape == (pred.n_obs, 3)
    assert np.isfinite(pred.obsm["spatial_3D"]).all()

    output = OUT / name
    pred.write_h5ad(output)

    print("\nSaved:", output)
    print("Cells:", pred.n_obs)
    print("Radius of gyration:", round(radius(xyz), 2))
    print("Spatial extent:", np.round(np.ptp(xyz, axis=0), 2))
    print(
        "Unique positions:",
        len(np.unique(pred.obsm["spatial_3D"], axis=0))
    )

save_candidate("e775_v3_global.h5ad", xyz_global)
save_candidate("e775_v3_tissue.h5ad", xyz_tissue)

# Compare the three geometry candidates.
fig = plt.figure(figsize=(16, 5))

models = [
    ("V2: uniform 0.75", np.asarray(v2.obsm["spatial_3D"])),
    ("V3: global log-scale", xyz_global),
    ("V3: tissue-aware", xyz_tissue),
]

sample = np.random.default_rng(123).choice(
    v2.n_obs, size=min(6000, v2.n_obs),
    replace=False
)

for i, (name, xyz) in enumerate(models):
    ax = fig.add_subplot(1, 3, i + 1, projection="3d")
    ax.scatter(
        xyz[sample, 0],
        xyz[sample, 1],
        xyz[sample, 2],
        s=0.4,
        alpha=0.3
    )
    ax.set_title(name)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect(np.maximum(np.ptp(xyz, axis=0), 1))

plt.tight_layout()

figure = OUT / "e775_v3_geometry_comparison.png"
plt.savefig(figure, dpi=200, bbox_inches="tight")
print("\nSaved comparison figure:", figure)

plt.show()
print("\nV3 candidate generation complete.")
