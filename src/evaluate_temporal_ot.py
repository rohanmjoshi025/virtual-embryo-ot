from pathlib import Path 
import numpy as np 
import pandas as pd
import scanpy as sc
from scipy import sparse 
from moscot.problems.time import TemporalProblem

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

# Load the same pilot dataset
adata = sc.read_h5ad(RESULTS / "embryo_pilot.h5ad")

# Reproduce the RNA-only baseline
problem = TemporalProblem(adata)

problem = problem.prepare(
    time_key="time",
    joint_attr="X_pca",
    policy="sequential",
    a=False,
    b=False,
)

problem = problem.solve(
    epsilon=0.05,
    max_iterations=2000,
)

for (t1, t2), subproblem in problem.problems.items():

    P = subproblem.solution.transport_matrix

    if sparse.issparse(P):
        P = P.toarray()
    else:
        P = np.asarray(P)

    source = subproblem.adata_src
    target = subproblem.adata_tgt

    source_types = source.obs["celltype"].astype(str).to_numpy()
    target_types = target.obs["celltype"].astype(str).to_numpy()

    # Calculate total transport mass by cell type
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
        P,
    )

    # Normalize to source-type percentages
    row_totals = mass.sum(axis=1, keepdims=True)

    percentages = np.divide(
        mass,
        row_totals,
        out=np.zeros_like(mass),
        where=row_totals > 0,
    ) * 100

    transitions = pd.DataFrame(
        percentages,
        index=source_names,
        columns=target_names,
    )

    # Target population frequencies
    target_frequencies = (
        target.obs["celltype"]
        .astype(str)
        .value_counts(normalize=True)
    )

    enrichment_results = []

    for celltype in transitions.index:

        if celltype not in transitions.columns:
            continue

        observed = transitions.loc[celltype, celltype] / 100
        expected = target_frequencies[celltype]

        enrichment_results.append({
            "celltype": celltype,
            "observed_percent": observed * 100,
            "expected_percent": expected * 100,
            "enrichment": observed / expected,
        })

    enrichment_df = pd.DataFrame(enrichment_results)
    enrichment_df = enrichment_df.sort_values(
        "enrichment", ascending=False
    )

    print(f"\n===== E{t1} -> E{t2} =====")
    print(enrichment_df.round(3).to_string(index=False))

    enrichment_df.to_csv(
        RESULTS / f"enrichment_{t1}_{t2}.csv",
        index=False,
    )

print("\nEvaluation completed.")