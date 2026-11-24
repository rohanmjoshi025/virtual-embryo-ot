from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
OUT = ROOT / 'submission'
OUT.mkdir(exist_ok=True)
SEED = 42
N = 4000
ALPHA = 0.5  # E7.75 is the midpoint between E7.5 and E8.0
rng = np.random.default_rng(SEED)


def dense(matrix):
    return (matrix.toarray() if hasattr(matrix, 'toarray') else np.asarray(matrix)).astype(np.float32)


def radius(xyz):
    centered = xyz - xyz.mean(axis=0)
    return float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))


def axes(xyz):
    cov = np.cov(xyz, rowvar=False)
    values, vectors = np.linalg.eigh(cov)
    order = np.argsort(values)[::-1]
    return np.sqrt(np.maximum(values[order], 0)), vectors[:, order]


def impose_axis_spreads(xyz, desired_stds):
    center = xyz.mean(axis=0)
    _, vectors = axes(xyz)
    projected = (xyz - center) @ vectors
    current_stds = projected.std(axis=0, ddof=1)
    if np.any(current_stds < 1e-8):
        raise ValueError('Degenerate spatial coordinate axis')
    projected *= desired_stds / current_stds
    return (projected @ vectors.T + center).astype(np.float32)


print('Loading the released E7.5 and training E8.0 embryos...')
a = sc.read_h5ad(DATA / 'E7.5.h5ad')
b = sc.read_h5ad(DATA / 'E8.0.h5ad')

# The final panel is NOT the validation panel. Prefer the published file.
final_panel_file = OUT / 'T2__embryo__test_interp.genes.txt'
if final_panel_file.exists():
    panel = [g.strip() for g in final_panel_file.read_text().splitlines() if g.strip()]
    print('Loaded official panel from:', final_panel_file)
else:
    # This matches the published test panel as of Nov 2026,
    # but the official format checker is still authoritative.
    omissions = {'Casp4', 'Pnliprp1', 'AI646519'}
    panel = [g for g in b.var_names if g not in omissions]
    print('Using provisional 497-gene panel; verify against official file!')

assert len(panel) == 497 and len(set(panel)) == 497
assert set(panel) <= set(a.var_names)
assert set(panel) <= set(b.var_names)

# Use distinct target cells to avoid duplicated XYZ positions.
late_indices = rng.choice(b.n_obs, size=N, replace=False)
X_early = dense(a[:, panel].X)
X_late = dense(b[:, panel].X)[late_indices]

print('Fitting RNA PCA and finding five E7.5 neighbors per E8.0 cell...')
features = np.vstack([X_early, X_late])
features = StandardScaler().fit_transform(features)
Z = PCA(n_components=30, random_state=SEED).fit_transform(features)
za, zb = Z[:a.n_obs], Z[a.n_obs:]

dist, idx = NearestNeighbors(n_neighbors=5).fit(za).kneighbors(zb)
sigma = max(float(np.median(dist[:, 0])), 1e-6)
energy = dist**2
weights = np.exp(-(energy - energy.min(axis=1, keepdims=True)) / (2 * sigma**2))
weights /= weights.sum(axis=1, keepdims=True)

X_early_soft = np.sum(X_early[idx] * weights[:, :, None], axis=1)
X_soft = ((1 - ALPHA) * X_early_soft + ALPHA * X_late).astype(np.float32)

# One sampled early neighbor, to preserve expression heterogeneity.
choices = (rng.random(N)[:, None] > np.cumsum(weights, axis=1)).sum(axis=1)
choices = np.clip(choices, 0, 4)
X_early_hard = X_early[idx[np.arange(N), choices]]
linear = ((1 - ALPHA) * np.expm1(X_early_hard) + ALPHA * np.expm1(X_late))
linear = np.maximum(linear, 0)
linear *= 10000.0 / np.maximum(linear.sum(axis=1, keepdims=True), 1e-8)
X_hard = np.log1p(linear).astype(np.float32)
X_blend = (0.85 * X_soft + 0.15 * X_hard).astype(np.float32)

print('Constructing two E7.75 geometry hypotheses...')
xyz_early_full = np.asarray(a.obsm['spatial_3D'], dtype=np.float64)
xyz_late_full = np.asarray(b.obsm['spatial_3D'], dtype=np.float64)
xyz_late = xyz_late_full[late_indices].copy()
std_early, _ = axes(xyz_early_full)
std_late, _ = axes(xyz_late_full)
target_std = (1 - ALPHA) * std_early + ALPHA * std_late
xyz_pca = impose_axis_spreads(xyz_late, target_std)

# A control with the same overall size but unchanged E8.0 shape.
rg_goal = (radius(xyz_early_full) ** (1 - ALPHA) *
           radius(xyz_late_full) ** ALPHA)
xyz_iso = (xyz_late.mean(axis=0) +
           (rg_goal / radius(xyz_late)) * (xyz_late - xyz_late.mean(axis=0)))
xyz_iso = xyz_iso.astype(np.float32)


def save(name, expression, coords):
    pred = ad.AnnData(
        X=expression.astype(np.float32),
        obs=pd.DataFrame(index=[f'pred_{i:05d}' for i in range(N)]),
        var=pd.DataFrame(index=panel),
    )
    pred.obsm['spatial_3D'] = coords.astype(np.float32)
    assert pred.shape == (N, 497)
    assert np.isfinite(pred.X).all() and (pred.X >= 0).all()
    assert np.isfinite(pred.obsm['spatial_3D']).all()
    output = OUT / f'{name}.h5ad'
    pred.write_h5ad(output)
    print(f'{name}: {pred.shape}, Rg={radius(coords):.2f}, unique XYZ={np.unique(coords, axis=0).shape[0]}')
    return output


save('e775_v6_pca_soft', X_soft, xyz_pca)
save('e775_v6_pca_blend15', X_blend, xyz_pca)
save('e775_v6_iso_soft', X_soft, xyz_iso)

# No artificial RNA mixing: retain E7.5 expression-position coherence.
early_indices = rng.choice(a.n_obs, size=N, replace=False)
xyz_early = np.asarray(a.obsm['spatial_3D'], dtype=np.float64)[early_indices]
xyz_early_scaled = impose_axis_spreads(xyz_early, target_std)
save('e775_v6_earlycopy', X_early[early_indices], xyz_early_scaled)

print('E7.5 PCA spreads:', np.round(std_early, 2))
print('E8.0 PCA spreads:', np.round(std_late, 2))
print('Predicted E7.75 PCA spreads:', np.round(target_std, 2))
print('Ready for the official 497-gene test-board FORMAT checker, not yet scored.')
