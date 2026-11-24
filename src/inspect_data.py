from pathlib import Path
import scanpy as sc

data_dir = Path("data")

stages = ["E6.75", "E7.25", "E8.0"]

for stage in stages:
    path = data_dir / f"{stage}.h5ad"

    if not path.exists():
        print(f"Missing: {path}")
        continue

    print(f"\n{'=' * 50}")
    print(f"STAGE: {stage}")
    print(f"{'=' * 50}")

    # Backed mode avoids loading the full matrix into RAM
    adata = sc.read_h5ad(path, backed="r")

    print("Number of cells:", adata.n_obs)
    print("Number of genes:", adata.n_vars)
    print("Expression shape:", adata.shape)

    print("\nCell metadata:")
    print(adata.obs.columns.tolist())

    print("\nGene metadata:")
    print(adata.var.columns.tolist())

    print("\nSpatial coordinate keys:")
    print(list(adata.obsm.keys()))

    if "spatial_3D" in adata.obsm:
        coords = adata.obsm["spatial_3D"]
        print("3D coordinate shape:", coords.shape)

    print("\nFirst 10 genes:")
    print(adata.var_names[:10].tolist())

    adata.file.close()
