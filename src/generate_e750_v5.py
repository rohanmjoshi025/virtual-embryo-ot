from pathlib import Path
import numpy as np
import scanpy as sc

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "submission"

v2 = sc.read_h5ad(
    OUT / "e750_validation_v2.h5ad"
)
hard = sc.read_h5ad(
    OUT / "e750_v4_hard_linear.h5ad"
)

assert v2.shape == hard.shape
assert list(v2.var_names) == list(hard.var_names)
assert np.allclose(
    v2.obsm["spatial_3D"],
    hard.obsm["spatial_3D"]
)

def dense(X):
    if hasattr(X, "toarray"):
        return X.toarray().astype(np.float32)
    return np.asarray(X, dtype=np.float32)

X_v2 = dense(v2.X)
X_hard = dense(hard.X)

for weight in [0.15, 0.30, 0.50]:
    candidate = v2.copy()

    X = (
        (1 - weight) * X_v2
        + weight * X_hard
    ).astype(np.float32)

    candidate.X = X

    assert np.isfinite(X).all()
    assert (X >= 0).all()

    name = f"e750_v5_blend_{int(weight*100)}"
    path = OUT / f"{name}.h5ad"

    candidate.write_h5ad(path)

    linear = np.expm1(X)
    library = linear.sum(axis=1)

    print(f"\n{name}")
    print("Shape:", candidate.shape)
    print("Mean expression:", round(float(X.mean()), 4))
    print("Fraction zero:", round(float((X == 0).mean()), 4))
    print("Median implied library:",
          round(float(np.median(library)), 2))
    print("Saved:", path)

print("\nAll V5 candidates generated.")
