"""GraphSAGE edge classifier on the WEB-IDS23 temporal graph.

Multi-seed runner that writes per-seed and summary results in the same
format used by anomaly_detection.py.

The temporal graph is built in-process from the merged CSV so we are
not blocked by the label-parsing issue in build_temporal_graph.py.
The build seed is fixed (BUILD_SEED) so all model-training seeds see
the same supervised graph view; only model init varies across seeds.
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

from anomaly_detection import (
    evaluate_scores,
    print_seed_result,
    save_seed_result,
    save_summary,
    set_seed,
)


INPUT_FILE = "web_ids23_merged_clean.csv"
GRAPH_FILE = "graphs/web_ids23_temporal_graph_supervised.pt"
RESULTS_DIR = "results_temporal"
MODEL_NAME = "graphsage"
DEFAULT_SEEDS = "42,7,13,21,100"

BUILD_SEED = 42
FLOW_SUBSAMPLE = 400_000
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.3
LR = 5e-3
WEIGHT_DECAY = 5e-4
EPOCHS = 60
PATIENCE = 10
EDGE_BATCH_SIZE = 65_536
THRESHOLD_PERCENTILE = 95.0

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

NODE_AGG_COLUMNS = [
    "fwd_pkts_tot",
    "bwd_pkts_tot",
    "fwd_data_pkts_tot",
    "bwd_data_pkts_tot",
    "flow_duration",
    "payload_bytes_per_second",
]


def parse_labels(series):
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        return numeric.fillna(0).astype(int)
    return series.astype(str).str.strip().str.lower().eq("attack").astype(int)


def stratified_indices(labels, target_size, rng):
    labels = np.asarray(labels)
    benign = np.flatnonzero(labels == 0)
    attack = np.flatnonzero(labels == 1)
    benign_share = len(benign) / len(labels)
    attack_share = len(attack) / len(labels)
    benign_take = min(len(benign), int(round(target_size * benign_share)))
    attack_take = min(len(attack), int(round(target_size * attack_share)))
    benign_pick = rng.choice(benign, size=benign_take, replace=False)
    attack_pick = rng.choice(attack, size=attack_take, replace=False)
    indices = np.concatenate([benign_pick, attack_pick])
    rng.shuffle(indices)
    return indices


def temporal_features(df, time_column):
    ts = df[time_column].fillna(0).to_numpy(dtype=np.float64)
    ts_min, ts_max = ts.min(), ts.max()
    ts_range = ts_max - ts_min if ts_max > ts_min else 1.0
    time_norm = (ts - ts_min) / ts_range

    df_temp = pd.DataFrame({"ts": ts, "src": df["id.orig_h"].to_numpy()})
    df_temp["prev"] = df_temp.groupby("src")["ts"].shift(1)
    inter_arrival = (df_temp["ts"] - df_temp["prev"]).fillna(0.0).to_numpy()
    inter_arrival = np.clip(inter_arrival, 0.0, None)

    hour = (ts % 86400) / 3600.0
    hour_sin = np.sin(2.0 * np.pi * hour / 24.0)
    hour_cos = np.cos(2.0 * np.pi * hour / 24.0)
    return np.stack([time_norm, inter_arrival, hour_sin, hour_cos], axis=1).astype(np.float32)


def build_node_features(df, all_nodes, train_mask):
    train_df = df.loc[train_mask]
    src_stats = train_df.groupby("id.orig_h")[NODE_AGG_COLUMNS].agg(["sum", "mean", "std"])
    dst_stats = train_df.groupby("id.resp_h")[NODE_AGG_COLUMNS].agg(["sum", "mean", "std"])
    combined = src_stats.add(dst_stats, fill_value=0)
    combined.columns = ["_".join(col) for col in combined.columns]

    node_features = []
    empty = [0.0] * combined.shape[1]
    for ip in all_nodes:
        if ip in combined.index:
            node_features.append(combined.loc[ip].values)
        else:
            node_features.append(empty)

    node_features = np.array(node_features, dtype=np.float32)
    node_features = np.nan_to_num(node_features, nan=0.0, posinf=0.0, neginf=0.0)
    scaler = StandardScaler()
    node_features = scaler.fit_transform(node_features)
    return np.nan_to_num(node_features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def build_temporal_data():
    rng = np.random.default_rng(BUILD_SEED)

    needed = ["id.orig_h", "id.resp_h", "attack", *EDGE_FEATURE_COLUMNS]
    available = pd.read_csv(INPUT_FILE, nrows=0).columns
    time_column = "ts_epoch" if "ts_epoch" in available else "ts"
    needed.append(time_column)

    print(f"Loading {INPUT_FILE} ...")
    df = pd.read_csv(INPUT_FILE, usecols=needed, low_memory=False)
    df["attack"] = parse_labels(df["attack"])
    for col in EDGE_FEATURE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[time_column] = pd.to_numeric(df[time_column], errors="coerce")
    df = df.dropna(subset=["id.orig_h", "id.resp_h"]).reset_index(drop=True)

    keep = stratified_indices(df["attack"].to_numpy(), FLOW_SUBSAMPLE, rng)
    keep.sort()
    df = df.iloc[keep].reset_index(drop=True)

    all_nodes = pd.concat([df["id.orig_h"], df["id.resp_h"]]).unique()
    node_to_idx = {ip: i for i, ip in enumerate(all_nodes)}
    src = df["id.orig_h"].map(node_to_idx).to_numpy()
    dst = df["id.resp_h"].map(node_to_idx).to_numpy()

    labels = df["attack"].to_numpy()
    indices = np.arange(len(labels))
    train_idx, temp_idx = train_test_split(
        indices, train_size=0.70, random_state=BUILD_SEED,
        stratify=labels, shuffle=True,
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, random_state=BUILD_SEED,
        stratify=labels[temp_idx], shuffle=True,
    )
    train_mask = np.zeros(len(labels), dtype=bool)
    val_mask = np.zeros(len(labels), dtype=bool)
    test_mask = np.zeros(len(labels), dtype=bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    flow_features = df[EDGE_FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    flow_features = np.nan_to_num(flow_features, nan=0.0, posinf=0.0, neginf=0.0)
    edge_scaler = StandardScaler()
    edge_scaler.fit(flow_features[train_mask])
    flow_features = np.nan_to_num(
        edge_scaler.transform(flow_features), nan=0.0, posinf=0.0, neginf=0.0,
    ).astype(np.float32)

    temp_features = temporal_features(df, time_column)
    temp_scaler = StandardScaler()
    temp_scaler.fit(temp_features[train_mask])
    temp_features = np.nan_to_num(
        temp_scaler.transform(temp_features), nan=0.0, posinf=0.0, neginf=0.0,
    ).astype(np.float32)

    edge_attr = np.concatenate([flow_features, temp_features], axis=1)
    node_features = build_node_features(df, all_nodes, train_mask)

    flow_edge_index = np.vstack([src, dst])
    pair_keys = src.astype(np.int64) * len(all_nodes) + dst.astype(np.int64)
    _, unique_idx = np.unique(pair_keys, return_index=True)
    unique_src = src[unique_idx]
    unique_dst = dst[unique_idx]
    mp_edge_index = np.vstack([
        np.concatenate([unique_src, unique_dst]),
        np.concatenate([unique_dst, unique_src]),
    ])

    data = Data(
        x=torch.tensor(node_features, dtype=torch.float),
        edge_index=torch.tensor(mp_edge_index, dtype=torch.long),
        flow_edge_index=torch.tensor(flow_edge_index, dtype=torch.long),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float),
        edge_label=torch.tensor(labels, dtype=torch.long),
        edge_time=torch.tensor(
            df[time_column].fillna(0).to_numpy(), dtype=torch.float,
        ),
        train_mask=torch.tensor(train_mask),
        val_mask=torch.tensor(val_mask),
        test_mask=torch.tensor(test_mask),
    )
    print(f"  rows after subsample: {len(df):,}  attack share: {df['attack'].mean():.4f}")
    print(f"  nodes: {data.x.shape[0]:,}  mp edges: {data.edge_index.shape[1]:,}"
          f"  flow edges: {data.flow_edge_index.shape[1]:,}")

    os.makedirs(os.path.dirname(GRAPH_FILE), exist_ok=True)
    torch.save(data, GRAPH_FILE)
    print(f"  cached to {GRAPH_FILE}")
    return data


def load_or_build():
    if os.path.exists(GRAPH_FILE):
        print(f"Reusing cached graph at {GRAPH_FILE}")
        return torch.load(GRAPH_FILE, weights_only=False)
    return build_temporal_data()


class SAGEEdgeClassifier(nn.Module):
    def __init__(self, node_in, edge_in, hidden_dim, num_layers, dropout):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(node_in, hidden_dim, aggr="mean"))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim, aggr="mean"))
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def encode(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def classify(self, h, flow_edge_index, edge_attr):
        src, dst = flow_edge_index[0], flow_edge_index[1]
        feats = torch.cat([h[src], h[dst], edge_attr], dim=1)
        return self.edge_mlp(feats).squeeze(-1)


def edge_scores(model, data, indices, batch_size):
    h = model.encode(data.x, data.edge_index)
    probs = []
    labels = []
    for start in range(0, len(indices), batch_size):
        batch = indices[start:start + batch_size]
        flow_edges = data.flow_edge_index[:, batch]
        attrs = data.edge_attr[batch]
        logits = model.classify(h, flow_edges, attrs)
        probs.append(torch.sigmoid(logits).detach().cpu().numpy())
        labels.append(data.edge_label[batch].cpu().numpy())
    return np.concatenate(probs), np.concatenate(labels)


def train_one_seed(data, seed, device, threshold_percentile):
    set_seed(seed)

    train_idx = data.train_mask.nonzero(as_tuple=False).flatten()
    val_idx = data.val_mask.nonzero(as_tuple=False).flatten()
    test_idx = data.test_mask.nonzero(as_tuple=False).flatten()

    train_labels = data.edge_label[train_idx]
    pos = (train_labels == 1).sum().item()
    neg = (train_labels == 0).sum().item()
    pos_weight = torch.tensor([neg / max(pos, 1)], device=device)

    model = SAGEEdgeClassifier(
        node_in=data.x.shape[1],
        edge_in=data.edge_attr.shape[1],
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    rng = np.random.default_rng(seed)
    best_val_auc = -np.inf
    best_state = None
    epochs_since_best = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        perm = rng.permutation(len(train_idx))
        train_idx_shuffled = train_idx[torch.tensor(perm, device=device)]

        epoch_loss_sum = 0.0
        epoch_count = 0
        for start in range(0, len(train_idx_shuffled), EDGE_BATCH_SIZE):
            batch = train_idx_shuffled[start:start + EDGE_BATCH_SIZE]
            optimizer.zero_grad()
            h = model.encode(data.x, data.edge_index)
            flow_edges = data.flow_edge_index[:, batch]
            attrs = data.edge_attr[batch]
            logits = model.classify(h, flow_edges, attrs)
            target = data.edge_label[batch].float()
            loss = loss_fn(logits, target)
            loss.backward()
            optimizer.step()
            epoch_loss_sum += loss.item() * len(batch)
            epoch_count += len(batch)
        train_loss = epoch_loss_sum / epoch_count

        model.eval()
        with torch.no_grad():
            val_probs, val_labels_np = edge_scores(model, data, val_idx, EDGE_BATCH_SIZE)
        val_eval = evaluate_scores(
            val_labels_np, val_probs, threshold_percentile=threshold_percentile,
        )

        if val_eval["roc_auc"] > best_val_auc + 1e-4:
            best_val_auc = val_eval["roc_auc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_since_best = 0
        else:
            epochs_since_best += 1

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"  epoch {epoch:3d} | train_loss {train_loss:.4f}"
                f" | val_roc_auc {val_eval['roc_auc']:.4f}"
                f" | val_recall {val_eval['attack_recall']:.4f}"
            )

        if epochs_since_best >= PATIENCE:
            print(f"  early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        val_probs, val_labels_np = edge_scores(model, data, val_idx, EDGE_BATCH_SIZE)
        test_probs, test_labels_np = edge_scores(model, data, test_idx, EDGE_BATCH_SIZE)
    val_metrics = evaluate_scores(
        val_labels_np, val_probs, threshold_percentile=threshold_percentile,
    )
    test_metrics = evaluate_scores(
        test_labels_np, test_probs, threshold=val_metrics["threshold"],
    )
    return {"validation": val_metrics, "test": test_metrics}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-file", default=GRAPH_FILE)
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    parser.add_argument("--seeds", default=DEFAULT_SEEDS)
    parser.add_argument(
        "--threshold-percentile",
        type=float,
        default=THRESHOLD_PERCENTILE,
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    device = torch.device(args.device)

    data = load_or_build()
    data = data.to(device)
    print(f"Nodes: {data.x.shape[0]:,}  MP edges: {data.edge_index.shape[1]:,}"
          f"  Flow edges: {data.flow_edge_index.shape[1]:,}")
    train_pos = int((data.edge_label[data.train_mask] == 1).sum().item())
    train_neg = int((data.edge_label[data.train_mask] == 0).sum().item())
    print(f"Train pos/neg: {train_pos}/{train_neg}")
    print(f"Device: {device}")
    print(f"Seeds: {seeds}")
    print(f"Results dir: {args.results_dir}")

    config = {
        "graph_file": args.graph_file,
        "model": MODEL_NAME,
        "task": "edge_classification",
        "build_seed": BUILD_SEED,
        "flow_subsample": FLOW_SUBSAMPLE,
        "hidden_dim": HIDDEN_DIM,
        "num_layers": NUM_LAYERS,
        "dropout": DROPOUT,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "epochs": EPOCHS,
        "patience": PATIENCE,
        "edge_batch_size": EDGE_BATCH_SIZE,
        "threshold_percentile": args.threshold_percentile,
        "device": str(device),
        "train_split_size": int(data.train_mask.sum().item()),
        "val_split_size": int(data.val_mask.sum().item()),
        "test_split_size": int(data.test_mask.sum().item()),
    }

    results = []
    for seed in seeds:
        print(f"\n\n######## Seed {seed} ########")
        result = train_one_seed(data, seed, device, args.threshold_percentile)
        results.append(result)
        path = save_seed_result(args.results_dir, MODEL_NAME, seed, result, config)
        print_seed_result(MODEL_NAME, seed, result["validation"], result["test"])
        print(f"Saved run: {path}")

    summary_path = save_summary(args.results_dir, MODEL_NAME, results)
    print(f"\nSaved summary: {summary_path}")


if __name__ == "__main__":
    main()
