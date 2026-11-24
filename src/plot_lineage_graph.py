
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
import networkx as nx
import matplotlib.pyplot as plt
from scipy import sparse
from moscot.problems.time import TemporalProblem

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

adata = sc.read_h5ad(RESULTS / "embryo_pilot.h5ad")

# Reproduce the existing RNA-only model
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

G = nx.DiGraph()
threshold = 10.0

for (t1, t2), subproblem in problem.problems.items():

    P = subproblem.solution.transport_matrix

    if sparse.issparse(P):
        P = P.toarray()
    else:
        P = np.asarray(P)

    source_types = (
        subproblem.adata_src.obs["celltype"]
        .astype(str).to_numpy()
    )

    target_types = (
        subproblem.adata_tgt.obs["celltype"]
        .astype(str).to_numpy()
    )

    # Aggregate transport mass by annotated cell type
    source_names, source_codes = np.unique(
        source_types, return_inverse=True
    )
    target_names, target_codes = np.unique(
        target_types, return_inverse=True
    )

    source_indicator = np.eye(len(source_names))[source_codes]
    target_indicator = np.eye(len(target_names))[target_codes]

    mass = source_indicator.T @ P @ target_indicator

    totals = mass.sum(axis=1, keepdims=True)

    percentages = np.divide(
        mass,
        totals,
        out=np.zeros_like(mass),
        where=totals > 0,
    ) * 100

    for i, source_name in enumerate(source_names):
        for j, target_name in enumerate(target_names):

            pct = percentages[i, j]

            if pct < threshold:
                continue

            source_node = f"E{t1}: {source_name}"
            target_node = f"E{t2}: {target_name}"

            G.add_edge(
                source_node,
                target_node,
                weight=float(pct),
            )

# Arrange nodes in columns by developmental stage
stages = [6.75, 7.25, 8.0]
pos = {}

for x, stage in enumerate(stages):

    nodes = sorted(
        n for n in G.nodes
        if n.startswith(f"E{stage}:")
    )

    for y, node in enumerate(nodes):
        pos[node] = (x, -y)

plt.figure(figsize=(19, 13))

nx.draw_networkx_nodes(
    G,
    pos,
    node_size=450,
    node_color="lightblue",
)

nx.draw_networkx_edges(
    G,
    pos,
    arrows=True,
    arrowsize=12,
    width=[G[u][v]["weight"] / 10 for u, v in G.edges],
    alpha=0.45,
)

nx.draw_networkx_labels(
    G,
    pos,
    font_size=7,
)

plt.title("Inferred Cell-Type Transport Across Embryonic Stages")
plt.axis("off")
plt.tight_layout()

plt.savefig(
    RESULTS / "lineage_transport_graph.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()

print("Saved results/lineage_transport_graph.png")
