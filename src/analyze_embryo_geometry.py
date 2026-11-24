from pathlib import Path
import numpy as np
import scanpy as sc

ROOT = Path(__file__).resolve().parent.parent

files = {
    "E6.75": ROOT / "data/E6.75.h5ad",
    "E7.25": ROOT / "data/E7.25.h5ad",
    "E8.0": ROOT / "data/E8.0.h5ad",
    "E7.75 V2": ROOT / "submission/e775_v2_geometry.h5ad",
}

for name, path in files.items():
    a = sc.read_h5ad(path)
    xyz = np.asarray(a.obsm["spatial_3D"], dtype=float)

    centered = xyz - xyz.mean(axis=0)

    covariance = np.cov(centered, rowvar=False)

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # Standard deviation along principal axes.
    axis_std = np.sqrt(np.maximum(eigenvalues, 0))

    # Radius of gyration.
    rg = np.sqrt(np.mean(np.sum(centered**2, axis=1)))

    print(f"\n{name}")
    print("Cells:", len(xyz))
    print("Principal-axis std:", np.round(axis_std, 2))
    print("Radius of gyration:", round(rg, 2))
    print("Major-axis direction:",
          np.round(eigenvectors[:, 0], 3))
