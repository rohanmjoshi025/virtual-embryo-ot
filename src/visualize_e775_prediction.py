
from pathlib import Path
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent

files = {
    "E7.25 observed": ROOT / "data/E7.25.h5ad",
    "E7.75 predicted": ROOT / "submission/e775_ot_baseline.h5ad",
    "E8.0 observed": ROOT / "data/E8.0.h5ad",
}

fig = plt.figure(figsize=(16, 5))

for k, (label, path) in enumerate(files.items()):
    adata = sc.read_h5ad(path)
    xyz = np.asarray(adata.obsm["spatial_3D"])

    ax = fig.add_subplot(1, 3, k + 1, projection="3d")
    ax.scatter(
        xyz[:, 0],
        xyz[:, 1],
        xyz[:, 2],
        s=0.4,
        alpha=0.35,
        rasterized=True,
    )

    ax.set_title(f"{label}\n{len(xyz):,} cells")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    extent = np.ptp(xyz, axis=0)
    ax.set_box_aspect(np.maximum(extent, 1))

    print(f"\n{label}")
    print("Cells:", len(xyz))
    print("Spatial extent:", np.round(extent, 2))
    print("Spatial standard deviation:",
          np.round(xyz.std(axis=0), 2))

plt.tight_layout()

output = ROOT / "submission/e775_geometry_comparison.png"
plt.savefig(output, dpi=200, bbox_inches="tight")
print("\nSaved:", output)
plt.show()
