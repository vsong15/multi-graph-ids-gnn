import os

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data


INPUT_FILE = "web_ids23_merged_clean.csv"
OUTPUT_FILE = "graphs/web_ids23_knn_graph.pt"
RANDOM_STATE = 42
K_NEIGHBORS = 10

# Flow-level columns used to build per-IP node feature vectors.
FLOW_NUMERIC_COLUMNS = [
    "flow_duration",
    "fwd_pkts_tot",
    "bwd_pkts_tot",
    "fwd_data_pkts_tot",
    "bwd_data_pkts_tot",
    "fwd_pkts_per_sec",
    "bwd_pkts_per_sec",
    "flow_pkts_per_sec",
    "down_up_ratio",
    "fwd_header_size_tot",
    "bwd_header_size_tot",
    "flow_FIN_flag_count",
    "flow_SYN_flag_count",
    "flow_RST_flag_count",
    "fwd_PSH_flag_count",
    "bwd_PSH_flag_count",
    "flow_ACK_flag_count",
    "fwd_URG_flag_count",
    "bwd_URG_flag_count",
    "flow_CWR_flag_count",
    "flow_ECE_flag_count",
    "payload_bytes_per_second",
    "fwd_init_window_size",
    "bwd_init_window_size",
]

# Aggregations applied per IP for each flow numeric column.
NODE_AGGS = ["sum", "mean", "std", "max"]


def load_data():
    needed = [
        "id.orig_h",
        "id.resp_h",
        "service",
        "attack",
        *FLOW_NUMERIC_COLUMNS,
    ]
    df = pd.read_csv(INPUT_FILE, usecols=needed, low_memory=False)

    # The merged CSV stores `attack` as the strings "attack"/"benign".
    attack_numeric = pd.to_numeric(df["attack"], errors="coerce")
    if attack_numeric.notna().any():
        df["attack"] = attack_numeric.fillna(0).astype(int)
    else:
        df["attack"] = (
            df["attack"].astype(str).str.strip().str.lower().eq("attack").astype(int)
        )
    for col in FLOW_NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["id.orig_h", "id.resp_h"]).reset_index(drop=True)
    df["service"] = df["service"].fillna("unknown").astype(str)
    return df


def build_node_table(df):
    # Concatenate src-side and dst-side rows so each flow contributes to both
    # endpoints' aggregates (frequency = number of flows the IP appears in).
    src_view = df.assign(ip=df["id.orig_h"], role="src")
    dst_view = df.assign(ip=df["id.resp_h"], role="dst")
    combined = pd.concat([src_view, dst_view], ignore_index=True)

    flow_stats = combined.groupby("ip")[FLOW_NUMERIC_COLUMNS].agg(NODE_AGGS)
    flow_stats.columns = ["_".join(col) for col in flow_stats.columns]

    role_counts = combined.groupby(["ip", "role"]).size().unstack(fill_value=0)
    role_counts = role_counts.reindex(columns=["src", "dst"], fill_value=0)
    role_counts.columns = ["src_flow_count", "dst_flow_count"]
    role_counts["total_flow_count"] = role_counts.sum(axis=1)

    service_diversity = (
        combined.groupby("ip")["service"].nunique().rename("service_diversity")
    )

    attack_ratio = combined.groupby("ip")["attack"].mean().rename("attack_ratio")
    attack_count = combined.groupby("ip")["attack"].sum().rename("attack_flow_count")

    nodes = flow_stats.join(role_counts).join(service_diversity)
    nodes = nodes.join(attack_ratio).join(attack_count)
    nodes = nodes.sort_index()

    # Majority vote: node is malicious if >=50% of its flows are attacks.
    nodes["label"] = (nodes["attack_ratio"] >= 0.5).astype(int)
    return nodes


def make_node_split(labels):
    labels = labels.to_numpy()
    indices = np.arange(len(labels))

    train_idx, temp_idx = train_test_split(
        indices,
        train_size=0.70,
        random_state=RANDOM_STATE,
        stratify=labels,
        shuffle=True,
    )
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=labels[temp_idx],
        shuffle=True,
    )

    train_mask = torch.zeros(len(labels), dtype=torch.bool)
    val_mask = torch.zeros(len(labels), dtype=torch.bool)
    test_mask = torch.zeros(len(labels), dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    return train_mask, val_mask, test_mask


def scale_features(features, train_mask):
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    scaler = StandardScaler()
    scaler.fit(features[train_mask.numpy()])
    features = scaler.transform(features)
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def build_knn_edges(features, k):
    # k+1 because each point's nearest neighbor is itself; we drop self below.
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", metric="euclidean")
    nn.fit(features)
    distances, indices = nn.kneighbors(features, return_distance=True)

    num_nodes = features.shape[0]
    src_rows = np.repeat(np.arange(num_nodes), k)
    dst_rows = indices[:, 1:].reshape(-1)
    dist_rows = distances[:, 1:].reshape(-1)

    # Make the graph undirected by adding reverse edges, then de-duplicate.
    src = np.concatenate([src_rows, dst_rows])
    dst = np.concatenate([dst_rows, src_rows])
    weights = np.concatenate([dist_rows, dist_rows])

    pairs = np.stack([src, dst], axis=1)
    pairs_sorted = np.sort(pairs, axis=1)
    _, unique_idx = np.unique(
        pairs_sorted[:, 0] * (num_nodes + 1) + pairs_sorted[:, 1],
        return_index=True,
    )
    src = src[unique_idx]
    dst = dst[unique_idx]
    weights = weights[unique_idx]

    edge_index = torch.tensor(np.vstack([src, dst]), dtype=torch.long)
    # Edge weight = exp(-d / median(d)) so that closer = stronger; useful for
    # GNNs that consume edge_attr.
    median_dist = np.median(weights[weights > 0]) if np.any(weights > 0) else 1.0
    edge_weight = np.exp(-weights / max(median_dist, 1e-9)).astype(np.float32)
    edge_attr = torch.tensor(edge_weight, dtype=torch.float).unsqueeze(1)
    return edge_index, edge_attr


def main():
    print("Loading merged dataset...")
    df = load_data()
    print(f"  rows: {len(df):,}")

    print("Building per-IP node feature table...")
    nodes = build_node_table(df)
    print(f"  unique IP nodes: {len(nodes):,}")

    feature_columns = [
        col
        for col in nodes.columns
        if col not in ("attack_ratio", "attack_flow_count", "label")
    ]
    raw_features = nodes[feature_columns].to_numpy(dtype=np.float32)
    labels = nodes["label"]

    train_mask, val_mask, test_mask = make_node_split(labels)
    features = scale_features(raw_features, train_mask)

    print(f"Computing k-NN graph (k={K_NEIGHBORS})...")
    edge_index, edge_attr = build_knn_edges(features, K_NEIGHBORS)

    data = Data(
        x=torch.tensor(features, dtype=torch.float),
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.tensor(labels.to_numpy(), dtype=torch.long),
    )
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    data.node_ids = list(nodes.index)
    data.feature_names = feature_columns

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    torch.save(data, OUTPUT_FILE)

    print("\nk-NN graph saved:")
    print(OUTPUT_FILE)

    print("\nGraph summary:\n")
    print(f"Nodes: {data.x.shape[0]:,}")
    print(f"Edges: {data.edge_index.shape[1]:,}")
    print(f"Node features: {data.x.shape[1]}")
    print(f"Edge features: {data.edge_attr.shape[1]}")

    print("\nLabel balance (node-level, majority vote):\n")
    unique, counts = data.y.unique(return_counts=True)
    for label, count in zip(unique, counts):
        pct = 100 * count.item() / data.y.numel()
        name = "Benign" if label.item() == 0 else "Attack"
        print(f"{name}: {count.item():,} ({pct:.2f}%)")

    print("\nSplit balance:\n")
    for name, mask in [
        ("Train", data.train_mask),
        ("Val", data.val_mask),
        ("Test", data.test_mask),
    ]:
        ys = data.y[mask]
        unique, counts = ys.unique(return_counts=True)
        summary = {int(l.item()): int(c.item()) for l, c in zip(unique, counts)}
        print(f"{name}: {mask.sum().item():,} nodes {summary}")

    print("\nNaN checks:\n")
    print("x has NaN:", torch.isnan(data.x).any().item())
    print("edge_attr has NaN:", torch.isnan(data.edge_attr).any().item())


if __name__ == "__main__":
    main()
