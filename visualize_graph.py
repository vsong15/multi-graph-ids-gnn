"""
Attack-pattern visualizations for the network intrusion detection graph.

Produces PNG figures in the visualizations/ directory:
  1. subgraph.png       — sampled subgraph; benign edges in blue, attack edges in red
  2. degree_dist.png    — log-scale degree distribution colored by attack rate
  3. top_attackers.png  — bar chart of top source IPs by attack flow count
  4. attack_types.png   — attack-type breakdown from the raw CSV (if available)
  5. score_dist.png     — anomaly score distribution (benign vs attack) per model,
                          loaded from results/ if present
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
import pandas as pd
import torch


GRAPH_FILE = "graphs/web_ids23_graph.pt"
CSV_FILE = "web_ids23_merged_clean.csv"
RESULTS_DIR = "results"
OUTPUT_DIR = "visualizations"

BENIGN_COLOR = "#4393c3"
ATTACK_COLOR = "#d6604d"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def load_graph(path):
    data = torch.load(path, weights_only=False, map_location="cpu")
    return data


def build_nx_subgraph(data, max_nodes=80, seed=42):
    """Sample the top-degree nodes and return a NetworkX DiGraph with attack labels."""
    rng = np.random.default_rng(seed)

    edge_index = data.edge_index.numpy()
    edge_label = data.edge_label.numpy()
    num_nodes = data.x.shape[0]

    # Compute out-degree per node and pick top-N
    degrees = np.bincount(edge_index[0], minlength=num_nodes)
    top_nodes = set(np.argsort(degrees)[-max_nodes:].tolist())

    # Collect edges where both endpoints are in the top-node set
    mask = np.array([
        (edge_index[0, i] in top_nodes and edge_index[1, i] in top_nodes)
        for i in range(edge_index.shape[1])
    ])

    sub_src = edge_index[0, mask]
    sub_dst = edge_index[1, mask]
    sub_labels = edge_label[mask]

    G = nx.DiGraph()
    G.add_nodes_from(top_nodes)
    for s, d, lbl in zip(sub_src, sub_dst, sub_labels):
        G.add_edge(int(s), int(d), attack=int(lbl))

    # Attach per-node attack rate
    node_attacks = {n: 0 for n in top_nodes}
    node_total = {n: 0 for n in top_nodes}
    for s, d, lbl in zip(sub_src, sub_dst, sub_labels):
        node_attacks[int(s)] += int(lbl)
        node_total[int(s)] += 1
        node_attacks[int(d)] += int(lbl)
        node_total[int(d)] += 1

    for n in G.nodes():
        total = node_total[n]
        G.nodes[n]["attack_rate"] = node_attacks[n] / total if total > 0 else 0.0

    return G


def plot_subgraph(G, out_path):
    fig, ax = plt.subplots(figsize=(14, 10))
    pos = nx.spring_layout(G, seed=42, k=0.6)

    attack_rates = [G.nodes[n]["attack_rate"] for n in G.nodes()]
    node_colors = [ATTACK_COLOR if r > 0.5 else BENIGN_COLOR for r in attack_rates]

    benign_edges = [(u, v) for u, v, d in G.edges(data=True) if d["attack"] == 0]
    attack_edges = [(u, v) for u, v, d in G.edges(data=True) if d["attack"] == 1]

    nx.draw_networkx_nodes(G, pos, node_size=60, node_color=node_colors, alpha=0.85, ax=ax)
    nx.draw_networkx_edges(
        G, pos, edgelist=benign_edges,
        edge_color=BENIGN_COLOR, alpha=0.35, width=0.8,
        arrows=True, arrowsize=8, ax=ax,
    )
    nx.draw_networkx_edges(
        G, pos, edgelist=attack_edges,
        edge_color=ATTACK_COLOR, alpha=0.7, width=1.2,
        arrows=True, arrowsize=10, ax=ax,
    )

    benign_patch = mpatches.Patch(color=BENIGN_COLOR, label="Benign flow")
    attack_patch = mpatches.Patch(color=ATTACK_COLOR, label="Attack flow")
    ax.legend(handles=[benign_patch, attack_patch], fontsize=11, loc="upper left")

    ax.set_title("Network communication subgraph (top active nodes)", fontsize=14)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_degree_distribution(data, out_path):
    edge_index = data.edge_index.numpy()
    edge_label = data.edge_label.numpy()
    num_nodes = data.x.shape[0]

    out_degree = np.bincount(edge_index[0], minlength=num_nodes)

    # Attack out-degree: count of attack edges leaving each node
    attack_mask = edge_label == 1
    attack_out = np.bincount(edge_index[0, attack_mask], minlength=num_nodes)
    attack_rate = np.where(out_degree > 0, attack_out / out_degree, 0.0)

    fig, ax = plt.subplots(figsize=(9, 5))

    degree_vals = out_degree[out_degree > 0]
    ar_vals = attack_rate[out_degree > 0]

    sc = ax.scatter(
        degree_vals, ar_vals,
        c=ar_vals, cmap="RdBu_r", alpha=0.6, s=15, vmin=0, vmax=1,
    )
    ax.set_xscale("log")
    ax.set_xlabel("Out-degree (log scale)", fontsize=12)
    ax.set_ylabel("Attack flow ratio", fontsize=12)
    ax.set_title("Node degree vs. attack flow ratio", fontsize=14)
    fig.colorbar(sc, ax=ax, label="Attack ratio")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_top_attackers(data, out_path, top_n=20):
    edge_index = data.edge_index.numpy()
    edge_label = data.edge_label.numpy()

    attack_mask = edge_label == 1
    attack_src = edge_index[0, attack_mask]

    src_counts = np.bincount(attack_src)
    top_idx = np.argsort(src_counts)[-top_n:][::-1]
    top_counts = src_counts[top_idx]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(
        range(len(top_idx)), top_counts,
        color=ATTACK_COLOR, alpha=0.85, edgecolor="white",
    )
    ax.set_xticks(range(len(top_idx)))
    ax.set_xticklabels([f"node {i}" for i in top_idx], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Attack flow count", fontsize=12)
    ax.set_title(f"Top {top_n} source nodes by attack flow count", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_attack_types(csv_path, out_path):
    if not os.path.exists(csv_path):
        print(f"CSV not found, skipping attack-type plot: {csv_path}")
        return

    df = pd.read_csv(csv_path, usecols=["attack", "attack_type"], low_memory=False)
    attack_df = df[df["attack"].astype(str) == "1"]

    if "attack_type" not in attack_df.columns or attack_df.empty:
        print("No attack_type column or no attacks found, skipping.")
        return

    counts = attack_df["attack_type"].value_counts()

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(counts)))
    ax.barh(counts.index[::-1], counts.values[::-1], color=colors[::-1], alpha=0.85, edgecolor="white")
    ax.set_xlabel("Number of flows", fontsize=12)
    ax.set_title("Attack-type distribution in dataset", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_score_distributions(results_dir, data, out_path):
    """Load per-seed result summaries and plot ROC-AUC comparison across models."""
    model_dirs = [
        d for d in os.listdir(results_dir)
        if os.path.isdir(os.path.join(results_dir, d))
    ]
    if not model_dirs:
        print("No results found, skipping score distribution plot.")
        return

    model_names = []
    test_roc = []
    test_pr = []
    test_recall = []

    for mname in sorted(model_dirs):
        summary_path = os.path.join(results_dir, mname, "summary.txt")
        if not os.path.exists(summary_path):
            continue
        with open(summary_path) as f:
            lines = f.read()

        def extract(metric, block):
            for line in lines.split("\n"):
                if block in lines[lines.find("Test:"):] and metric in line:
                    try:
                        return float(line.split(":")[1].split("+")[0].strip())
                    except Exception:
                        pass
            return None

        # Parse Test section
        test_section = lines[lines.find("Test:"):]
        roc = pr = recall = None
        for line in test_section.split("\n"):
            if "roc_auc:" in line:
                try:
                    roc = float(line.split(":")[1].split("+")[0].strip())
                except Exception:
                    pass
            elif "pr_auc:" in line:
                try:
                    pr = float(line.split(":")[1].split("+")[0].strip())
                except Exception:
                    pass
            elif "attack_recall:" in line:
                try:
                    recall = float(line.split(":")[1].split("+")[0].strip())
                except Exception:
                    pass

        if roc is not None:
            model_names.append(mname)
            test_roc.append(roc)
            test_pr.append(pr or 0.0)
            test_recall.append(recall or 0.0)

    if not model_names:
        print("Could not parse any model results, skipping.")
        return

    x = np.arange(len(model_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, test_roc, width, label="ROC-AUC", color="#4393c3", alpha=0.85)
    ax.bar(x, test_pr, width, label="PR-AUC", color="#92c5de", alpha=0.85)
    ax.bar(x + width, test_recall, width, label="Attack recall", color="#d6604d", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Test-set performance comparison across models", fontsize=14)
    ax.legend(fontsize=11)
    ax.axhline(0.9, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-file", default=GRAPH_FILE)
    parser.add_argument("--csv-file", default=CSV_FILE)
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--subgraph-nodes", type=int, default=80,
                        help="Number of top-degree nodes to include in the subgraph plot.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading graph from: {args.graph_file}")
    data = load_graph(args.graph_file)
    print(f"  Nodes: {data.x.shape[0]:,}  Edges: {data.edge_index.shape[1]:,}")

    print("\nBuilding subgraph...")
    G = build_nx_subgraph(data, max_nodes=args.subgraph_nodes)
    plot_subgraph(G, os.path.join(args.output_dir, "subgraph.png"))

    print("\nPlotting degree distribution...")
    plot_degree_distribution(data, os.path.join(args.output_dir, "degree_dist.png"))

    print("\nPlotting top attackers...")
    plot_top_attackers(data, os.path.join(args.output_dir, "top_attackers.png"))

    print("\nPlotting attack-type breakdown...")
    plot_attack_types(args.csv_file, os.path.join(args.output_dir, "attack_types.png"))

    print("\nPlotting model performance comparison...")
    plot_score_distributions(
        args.results_dir, data,
        os.path.join(args.output_dir, "model_comparison.png"),
    )

    print(f"\nAll visualizations saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
