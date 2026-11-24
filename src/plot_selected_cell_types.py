
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"

files = {
    "E6.75": DATA / "E6.75.h5ad",
    "E7.25": DATA / "E7.25.h5ad",
    "E8.0": DATA / "E8.0.h5ad",
}

selected_types = {
    "Primitive Streak": "red",
    "EXE-Endoderm": "blue",
    "Allantois": "green",
    "SOM": "orange",
}

fig = plt.figure(figsize=(18, 6))

for i, (stage, filename) in enumerate(files.items()):

    adata = sc.read_h5ad(filename)

    coords = np.asarray(
        adata.obsm["spatial_3D"]
    )

    labels = adata.obs["celltype"].astype(str).to_numpy()

    ax = fig.add_subplot(
        1, 3, i + 1,
        projection="3d"
    )

    # Background cells
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        coords[:, 2],
        color="lightgray",
        alpha=0.05,
        s=1,
    )

    # Highlight selected populations
    for celltype, color in selected_types.items():

        mask = labels == celltype

        if not np.any(mask):
            continue

        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            coords[mask, 2],
            color=color,
            s=3,
            alpha=0.7,
            label=celltype,
        )

    ax.set_title(stage)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.view_init(elev=20, azim=-60)

    ax.legend(
        loc="upper left",
        fontsize=7
    )

plt.tight_layout()

plt.savefig(
    RESULTS / "selected_celltype_spatial_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
