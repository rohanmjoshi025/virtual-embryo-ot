
import numpy as np
import scanpy as sc
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

path = ROOT / "submission" / "e775_ot_baseline_v1.h5ad"
reference_path = ROOT / "data" / "E8.0.h5ad"

pred = sc.read_h5ad(path)
reference = sc.read_h5ad(reference_path)


X = pred.X.toarray() if hasattr(pred.X, "toarray") else np.asarray(pred.X)
xyz = np.asarray(pred.obsm["spatial_3D"])

print("Prediction shape:", pred.shape)
print("Spatial shape:", xyz.shape)
print("Gene names unique:", pred.var_names.is_unique)
print("Panel matches E8.0:", list(pred.var_names) == list(reference.var_names))
print("Expression finite:", np.isfinite(X).all())
print("Expression nonnegative:", (X >= 0).all())
print("Coordinates finite:", np.isfinite(xyz).all())
print("Unique spatial positions:", len(np.unique(xyz, axis=0)))

assert pred.n_vars == 500
assert pred.var_names.is_unique
assert list(pred.var_names) == list(reference.var_names)
assert xyz.shape == (pred.n_obs, 3)
assert np.isfinite(X).all()
assert np.isfinite(xyz).all()
assert (X >= 0).all()

print("\nLocal structural validation passed.")
