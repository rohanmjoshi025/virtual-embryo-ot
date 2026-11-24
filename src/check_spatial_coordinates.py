
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

# Replace filenames with your actual dataset filenames.
files = {
    "E6.75": DATA / "E6.75.h5ad",
    "E7.25": DATA / "E7.25.h5ad",
    "E8.0": DATA / "E8.0.h5ad",
}

fig = plt.figure(figsize=(16, 5))

for index, (stage, filename) in enumerate(files.items()):

    adata = sc.read_h5ad(filename)

    coords = np.asarray(
        adata.obsm["spatial_3D"],
        dtype=float
    )

    print(f"\n===== {stage} =====")
    print("Number of cells:", coords.shape[0])
    print("Coordinate shape:", coords.shape)

    print("Minimum XYZ:", coords.min(axis=0))
    print("Maximum XYZ:", coords.max(axis=0))
    print("Mean XYZ:", coords.mean(axis=0))
    print("Standard deviation XYZ:", coords.std(axis=0))

    extent = np.ptp(coords, axis=0)
    print("Spatial extent XYZ:", extent)

    # Analyze the major spatial directions
    centered = coords - coords.mean(axis=0)
    covariance = np.cov(centered, rowvar=False)

    eigenvalues, eigenvectors = np.linalg.eigh(
        covariance
    )

    order = np.argsort(eigenvalues)[::-1]

    print(
        "Spatial eigenvalues:",
        eigenvalues[order]
    )

    print(
        "Major axis direction:",
        eigenvectors[:, order[0]]
    )

    # Plot embryo
    ax = fig.add_subplot(
        1, 3, index + 1,
        projection="3d"
    )

    
    celltypes = adata.obs["celltype"].astype("category")

    codes = celltypes.cat.codes.to_numpy()
    names = celltypes.cat.categories

    scatter = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        coords[:, 2],
        c=codes,
        cmap="tab20",
        s=2,
        alpha=0.5,
    )

    print(f"\n{stage} cell types:")
    for i, name in enumerate(names):
        print(f"{i}: {name}")


    ax.set_title(stage)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Preserve aspect ratio within each embryo
    ax.set_box_aspect(
        np.maximum(extent, 1e-8)
    )

plt.tight_layout()

plt.savefig(
    RESULTS / "spatial_coordinate_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# Add this inside your existing loop,
# after the coordinates have been loaded.

centroid = coords.mean(axis=0)
extent = np.ptp(coords, axis=0)

centered = coords - centroid

covariance = np.cov(centered, rowvar=False)

eigenvalues, eigenvectors = np.linalg.eigh(covariance)

order = np.argsort(eigenvalues)[::-1]

major_axis = eigenvectors[:, order[0]]

print(f"\n{stage} spatial diagnostics")
print("Centroid:", np.round(centroid, 2))
print("Extent XYZ:", np.round(extent, 2))
print("Major axis:", np.round(major_axis, 3))
print("Axis variances:", np.round(eigenvalues[order], 2))
