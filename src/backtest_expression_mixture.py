from pathlib import Path

import numpy as np
import scanpy as sc

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "submission"

SEED = 42
N = 4000

def dense(x):
    return (
        x.toarray() if hasattr(x, "toarray")
        else np.asarray(x)
    ).astype(np.float32)

print("Loading E7.25, E8.0 and V2...")

early = sc.read_h5ad(ROOT / "data/E7.25.h5ad")
late = sc.read_h5ad(ROOT / "data/E8.0.h5ad")
v2 = sc.read_h5ad(OUT / "e750_validation_v2.h5ad")

panel = list(v2.var_names)
assert len(panel) == 498
assert v2.n_obs == N

# Recover the E8.0 donor cells from V2.
rng = np.random.default_rng(SEED)
donors = rng.choice(late.n_obs, N, replace=False)

Xa = dense(early[:, panel].X)
Xb = dense(late[:, panel].X)[donors]

# Verify the saved V2 spatial donor selection.
xyz = np.asarray(
    late.obsm["spatial_3D"], dtype=np.float64
)[donors]

center = xyz.mean(axis=0)
reconstructed = center + 0.5881302486091871 * (
    xyz - center
)

assert np.allclose(
    reconstructed,
    v2.obsm["spatial_3D"],
    atol=0.02
), "Donor selection does not match V2."

print("Donor coordinates verified.")

# Match E8.0 donors to E7.25 RNA neighbors.
print("Finding RNA neighbors...")

features = np.vstack([Xa, Xb])
features = StandardScaler().fit_transform(features)

Z = PCA(
    n_components=30,
    random_state=SEED
).fit_transform(features)

Za = Z[:early.n_obs]
Zb = Z[early.n_obs:]

dist, idx = NearestNeighbors(
    n_neighbors=5
).fit(Za).kneighbors(Zb)

bandwidth = max(
    float(np.median(dist[:, 0])), 1e-6
)

weights = np.exp(
    -(dist**2 - dist.min(axis=1, keepdims=True)**2)
    / (2 * bandwidth**2)
)
weights /= weights.sum(axis=1, keepdims=True)

sample_rng = np.random.default_rng(123)
u = sample_rng.random(N)

selected_neighbor = (
    u[:, None] > np.cumsum(weights, axis=1)
).sum(axis=1)

selected_neighbor = np.clip(
    selected_neighbor, 0, 4
)

Xearly = Xa[
    idx[np.arange(N), selected_neighbor]
]

# All candidates retain exactly V2's geometry.
def save(name, expression):
    pred = v2.copy()
    pred.X = expression.astype(np.float32)

    assert pred.shape == (4000, 498)
    assert np.isfinite(pred.X).all()
    assert (pred.X >= 0).all()

    path = OUT / f"{name}.h5ad"
    pred.write_h5ad(path)

    print(
        name,
        "zeros:",
        round(float((pred.X == 0).mean()), 4),
        "saved:", path
    )

# Control: use real later-stage expression.
save("e750_mix_latecopy", Xb)

# Mixtures of observed, non-averaged profiles.
for late_fraction in [0.25, 0.50, 0.75]:
    mixture_rng = np.random.default_rng(
        SEED + int(late_fraction * 100)
    )

    choose_late = (
        mixture_rng.random(N) < late_fraction
    )

    Xpred = Xearly.copy()
    Xpred[choose_late] = Xb[choose_late]

    save(
        f"e750_mix_{int(late_fraction * 100)}",
        Xpred
    )

print("\nExpression mixture backtest complete.")
