
from pathlib import Path
import scanpy as sc
from moscot.problems.time import TemporalProblem

ROOT = Path(__file__).resolve().parent.parent

# Load the prepared dataset
adata = sc.read_h5ad(
    ROOT / "results" / "embryo_pilot.h5ad"
)

print("Dataset loaded:", adata.shape)
print("Developmental stages:")
print(adata.obs["time"].value_counts().sort_index())

# Initialize temporal optimal transport
problem = TemporalProblem(adata)

# Prepare sequential transport problems
problem = problem.prepare(
    time_key="time",
    joint_attr="X_pca",
    policy="sequential",
    a=False,
    b=False,
)

print("\nPrepared transport pairs:")
print(list(problem.problems.keys()))

# Solve optimal transport
problem = problem.solve(
    epsilon=0.05,
    max_iterations=2000,
)

print("\nSolver results:")

for pair, subproblem in problem.problems.items():
    print("\nStage pair:", pair)
    print("Solution shape:", subproblem.solution.shape)
    print("Converged:", subproblem.solution.converged)

print("\nTemporal OT fitting completed.")


import numpy as np
import pandas as pd
from scipy import sparse

# Select E6.75 -> E7.25
subproblem = problem.problems[(6.75, 7.25)]

# Extract the learned transport matrix
P = subproblem.solution.transport_matrix

if sparse.issparse(P):
    P = P.toarray()
else:
    P = np.asarray(P)

print("Transport matrix shape:", P.shape)

# Display a small section of the matrix
print("\nFirst 5 source cells x first 5 target cells:")
print(P[:5, :5])

# Check total mass and row sums
print("\nTotal transport mass:", P.sum())
print("First 5 row sums:", P.sum(axis=1)[:5])

# Show actual cell types
source = subproblem.adata_src
target = subproblem.adata_tgt

source_types = source.obs["celltype"].astype(str).to_numpy()
target_types = target.obs["celltype"].astype(str).to_numpy()

# Build a cell-type-level transport mass matrix
source_names, source_codes = np.unique(
    source_types, return_inverse=True
)
target_names, target_codes = np.unique(
    target_types, return_inverse=True
)

mass = np.zeros((len(source_names), len(target_names)))

np.add.at(
    mass,
    (source_codes[:, None], target_codes[None, :]),
    P
)

# Normalize each source cell type to percentages
row_totals = mass.sum(axis=1, keepdims=True)

percentages = np.divide(
    mass,
    row_totals,
    out=np.zeros_like(mass),
    where=row_totals > 0
) * 100

transition_table = pd.DataFrame(
    percentages,
    index=source_names,
    columns=target_names
)

print("\nCell-type transition percentages:")
print(transition_table.round(2).to_string())

# Save for later analysis
transition_table.to_csv(
    ROOT / "results" / "celltype_transition_percentages.csv"
)


# Compare same-type transport against target frequencies

target_frequencies = (
    target.obs["celltype"]
    .astype(str)
    .value_counts(normalize=True)
)

results = []

for celltype in transition_percentages.index:

    if celltype not in transition_percentages.columns:
        continue

    observed = transition_percentages.loc[
        celltype, celltype
    ] / 100

    expected = target_frequencies.get(celltype, 0)

    if expected > 0:
        enrichment = observed / expected
    else:
        enrichment = np.nan

    results.append({
        "celltype": celltype,
        "observed_percent": observed * 100,
        "expected_percent": expected * 100,
        "enrichment": enrichment,
    })

enrichment_table = pd.DataFrame(results)

print("\nSame-cell-type transport enrichment:")
print(
    enrichment_table
    .sort_values("enrichment", ascending=False)
    .round(3)
    .to_string(index=False)
)

enrichment_table.to_csv(
    RESULTS / f"enrichment_{t1}_{t2}.csv",
    index=False
)

small_table = pd.DataFrame(
    P[:5, :5],
    index=[
        f"{source.obs['celltype'].iloc[i]} (S{i+1})"
        for i in range(5)
    ],
    columns=[
        f"{target.obs['celltype'].iloc[j]} (T{j+1})"
        for j in range(5)
    ],
)

print("\nActual transport values:")
print(small_table.to_string())
