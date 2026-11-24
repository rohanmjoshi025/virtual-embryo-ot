from pathlib import Path
import scanpy as sc
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

print("Loading validation ground truth...")
truth = sc.read_h5ad(ROOT / "data/E7.5.h5ad")

print("Loading E7.25 reference...")
reference = sc.read_h5ad(ROOT / "data/E7.25.h5ad")

print("Loading V2 prediction...")
prediction = sc.read_h5ad(
    ROOT / "submission/e750_validation_v2.h5ad"
)

# Use the 498-gene panel accepted by the
# official E7.5 validation checker.
panel = list(reference.var_names)

assert len(panel) == 498
assert all(g in truth.var_names for g in panel)
assert all(g in prediction.var_names for g in panel)

# Align to identical gene ordering.
truth_aligned = truth[:, panel].copy()
prediction_aligned = prediction[:, panel].copy()

assert list(truth_aligned.var_names) == panel
assert list(prediction_aligned.var_names) == panel

# Keep original ground truth untouched.
output = ROOT / "data/E7.5_local_498.h5ad"
truth_aligned.write_h5ad(output)

def radius(xyz):
    centered = xyz - xyz.mean(axis=0)
    return np.sqrt(
        np.mean(np.sum(centered ** 2, axis=1))
    )

print("\n=== VALIDATION DATA ===")
print("Ground truth:", truth_aligned.shape)
print("Prediction:", prediction_aligned.shape)

for name, adata in [
    ("Truth", truth_aligned),
    ("V2", prediction_aligned)
]:
    X = (
        adata.X.toarray()
        if hasattr(adata.X, "toarray")
        else np.asarray(adata.X)
    )

    xyz = np.asarray(adata.obsm["spatial_3D"])

    print(f"\n{name}")
    print("Radius of gyration:", round(radius(xyz), 3))
    print("Fraction zero expression:", round(
        float(np.mean(X == 0)), 4
    ))
    print("Mean expression:", round(
        float(X.mean()), 4
    ))

print("\nSaved aligned ground truth:", output)
print("Ready for local scoring.")
