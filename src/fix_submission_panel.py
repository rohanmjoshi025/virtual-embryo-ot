
from pathlib import Path
import numpy as np
import scanpy as sc

ROOT = Path(__file__).resolve().parent.parent

INPUT = ROOT / "submission/e775_ot_baseline_v1.h5ad"
OUTPUT = ROOT / "submission/e775_ot_baseline_v1_498.h5ad"
REFERENCE = ROOT / "data/E7.25.h5ad"

# Use the 498-gene E7.25 panel as a candidate.
reference = sc.read_h5ad(REFERENCE)
panel = list(reference.var_names)

assert len(panel) == 498
assert len(set(panel)) == 498

# Load the existing 500-gene prediction.
pred = sc.read_h5ad(INPUT)

missing = [g for g in panel if g not in pred.var_names]
extra = [g for g in pred.var_names if g not in panel]

print("Original shape:", pred.shape)
print("Missing genes:", missing)
print("Extra genes:", extra)

assert not missing
assert set(extra) == {"Casp4", "Pnliprp1"}

# Select and reorder the expression columns.
corrected = pred[:, panel].copy()

xyz = np.asarray(corrected.obsm["spatial_3D"])
X = corrected.X
values = X.data if hasattr(X, "tocsr") else np.asarray(X)

assert corrected.n_vars == 498
assert xyz.shape == (corrected.n_obs, 3)
assert np.isfinite(values).all()
assert np.isfinite(xyz).all()
assert (values >= 0).all()

corrected.write_h5ad(OUTPUT)

# Verify the saved file.
check = sc.read_h5ad(OUTPUT)

assert check.shape == (3000, 498)
assert list(check.var_names) == panel

print("\nCorrected shape:", check.shape)
print("Spatial shape:", check.obsm["spatial_3D"].shape)
print("Saved:", OUTPUT)
print("Local validation passed.")

