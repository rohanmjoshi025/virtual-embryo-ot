from pathlib import Path
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent

files = {
    "E7.25 observed": ROOT / "data/E7.25.h5ad",
    "E7.75 predicted V2": ROOT / "submission/e775_v2_geometry.h5ad",
    "E8.0 observed": ROOT / "data/E8.0.h5ad",
}

fig = plt.figure(figsize=(17, 5))

for i, (name, path) in enumerate(files.items()):
    a = sc.read_h5ad(path)
    xyz = np.asarray(a.obsm["spatial_3D"])

    center = xyz.mean(axis=0)
    extent = np.ptp(xyz, axis=0)
    std = xyz.std(axis=0)

    tree = cKDTree(xyz)
    distances, _ = tree.query(xyz, k=2)
    nearest = distances[:, 1]

    print(f"\n{name}")
    print("Cells:", len(xyz))
    print("Spatial extent:", np.round(extent, 2))
    print("Spatial std:", np.round(std, 2))
    print("Median nearest-neighbor distance:",
          round(float(np.median(nearest)), 3))
    print("Unique positions:",
          len(np.unique(xyz, axis=0)))

    ax = fig.add_subplot(1, 3, i + 1, projection="3d")
    ax.scatter(
        xyz[:, 0],
        xyz[:, 1],
        xyz[:, 2],
        s=0.35,
        alpha=0.3,
        rasterized=True
    )

    ax.set_title(f"{name}\n{len(xyz):,} cells")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect(np.maximum(extent, 1))

plt.tight_layout()

output = ROOT / "submission/e775_v2_comparison.png"
plt.savefig(output, dpi=200, bbox_inches="tight")

print("\nSaved:", output)
plt.show()
