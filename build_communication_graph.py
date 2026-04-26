import os

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data


INPUT_FILE = "web_ids23_merged_clean.csv"
OUTPUT_FILE = "graphs/web_ids23_graph.pt"
RANDOM_STATE = 42

EDGE_FEATURE_COLUMNS = [
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
    "fwd_header_size_min",
    "fwd_header_size_max",
    "bwd_header_size_tot",
    "bwd_header_size_min",
    "bwd_header_size_max",
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
    "fwd_last_window_size",
    "bwd_last_window_size",
]

NODE_AGG_COLUMNS = [
    "fwd_pkts_tot",
    "bwd_pkts_tot",
    "fwd_data_pkts_tot",
    "bwd_data_pkts_tot",
    "flow_duration",
    "payload_bytes_per_second",
]


def make_split_masks(labels):
    labels = labels.to_numpy()
    benign_idx = np.flatnonzero(labels == 0)
    attack_idx = np.flatnonzero(labels == 1)

    train_benign, temp_benign = train_test_split(
        benign_idx,
        train_size=0.70,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    val_benign, test_benign = train_test_split(
        temp_benign,
        test_size=0.50,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    val_attack, test_attack = train_test_split(
        attack_idx,
        test_size=0.50,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    num_edges = len(labels)
    train_mask = torch.zeros(num_edges, dtype=torch.bool)
    val_mask = torch.zeros(num_edges, dtype=torch.bool)
    test_mask = torch.zeros(num_edges, dtype=torch.bool)

    train_mask[train_benign] = True
    val_mask[np.concatenate([val_benign, val_attack])] = True
    test_mask[np.concatenate([test_benign, test_attack])] = True

    return train_mask, val_mask, test_mask


def scale_features(features, train_mask):
    scaler = StandardScaler()
    train_rows = train_mask.numpy()

    features = np.nan_to_num(
        features,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    scaler.fit(features[train_rows])
    features = scaler.transform(features)

    return np.nan_to_num(
        features,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def build_node_features(df, all_nodes, train_mask):
    train_df = df.loc[train_mask.numpy()]

    src_stats = train_df.groupby("id.orig_h")[NODE_AGG_COLUMNS].agg(
        ["sum", "mean", "std"]
    )
    dst_stats = train_df.groupby("id.resp_h")[NODE_AGG_COLUMNS].agg(
        ["sum", "mean", "std"]
    )

    node_features_df = src_stats.add(dst_stats, fill_value=0)
    node_features_df.columns = [
        "_".join(col)
        for col in node_features_df.columns
    ]

    node_features = []
    empty_features = [0.0] * len(node_features_df.columns)

    for node in all_nodes:
        if node in node_features_df.index:
            node_features.append(node_features_df.loc[node].values)
        else:
            node_features.append(empty_features)

    node_features = np.array(node_features, dtype=np.float32)
    node_features = np.nan_to_num(
        node_features,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    scaler = StandardScaler()
    node_features = scaler.fit_transform(node_features)

    return np.nan_to_num(
        node_features,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


needed_columns = [
    "id.orig_h",
    "id.resp_h",
    "attack",
    *EDGE_FEATURE_COLUMNS,
]

available_columns = pd.read_csv(INPUT_FILE, nrows=0).columns
time_column = "ts_epoch" if "ts_epoch" in available_columns else "ts"
needed_columns.append(time_column)

df = pd.read_csv(INPUT_FILE, usecols=needed_columns, low_memory=False)

df["attack"] = pd.to_numeric(
    df["attack"],
    errors="coerce",
).fillna(0).astype(int)

for column in EDGE_FEATURE_COLUMNS:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce",
    )

df[time_column] = pd.to_numeric(
    df[time_column],
    errors="coerce",
)

df = df.dropna(
    subset=[
        "id.orig_h",
        "id.resp_h",
    ]
).reset_index(drop=True)

all_nodes = pd.concat([
    df["id.orig_h"],
    df["id.resp_h"],
]).unique()

node_to_idx = {
    node: i
    for i, node in enumerate(all_nodes)
}

src = df["id.orig_h"].map(node_to_idx).to_numpy()
dst = df["id.resp_h"].map(node_to_idx).to_numpy()

edge_index = torch.tensor(
    np.vstack([src, dst]),
    dtype=torch.long,
)

edge_labels = torch.tensor(
    df["attack"].to_numpy(),
    dtype=torch.long,
)

train_mask, val_mask, test_mask = make_split_masks(df["attack"])

edge_features = df[EDGE_FEATURE_COLUMNS].to_numpy(dtype=np.float32)
edge_features = scale_features(edge_features, train_mask)

node_features = build_node_features(
    df,
    all_nodes,
    train_mask,
)

data = Data(
    x=torch.tensor(node_features, dtype=torch.float),
    edge_index=edge_index,
    edge_attr=torch.tensor(edge_features, dtype=torch.float),
    edge_label=edge_labels,
)

data.train_mask = train_mask
data.val_mask = val_mask
data.test_mask = test_mask
data.edge_time = torch.tensor(
    df[time_column].fillna(0).to_numpy(),
    dtype=torch.float,
)

os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True,
)

torch.save(
    data,
    OUTPUT_FILE,
)

print("\nGraph saved:")
print(OUTPUT_FILE)

print("\nGraph summary:\n")
print(f"Nodes: {data.x.shape[0]}")
print(f"Edges / flows: {data.edge_index.shape[1]}")
print(f"Node features: {data.x.shape[1]}")
print(f"Edge features: {data.edge_attr.shape[1]}")

print("\nLabel balance:\n")
unique, counts = data.edge_label.unique(return_counts=True)
for label, count in zip(unique, counts):
    pct = 100 * count.item() / data.edge_label.numel()
    name = "Benign" if label.item() == 0 else "Attack"
    print(f"{name}: {count.item():,} ({pct:.2f}%)")

print("\nSplit balance:\n")
for name, mask in [
    ("Train", data.train_mask),
    ("Val", data.val_mask),
    ("Test", data.test_mask),
]:
    labels = data.edge_label[mask]
    unique, counts = labels.unique(return_counts=True)
    summary = {
        int(label.item()): int(count.item())
        for label, count in zip(unique, counts)
    }
    print(f"{name}: {mask.sum().item():,} edges {summary}")

print("\nNaN checks:\n")
print("x has NaN:", torch.isnan(data.x).any().item())
print("edge_attr has NaN:", torch.isnan(data.edge_attr).any().item())
