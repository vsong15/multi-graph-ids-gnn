"""GraphSAGE node classifier on the WEB-IDS23 k-NN graph.

Multi-seed runner that writes per-seed and summary results in the same
format used by anomaly_detection.py (results/<model>/seed_<n>.txt and
results/<model>/summary.txt).
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import SAGEConv

from anomaly_detection import (
    evaluate_scores,
    print_seed_result,
    save_seed_result,
    save_summary,
    set_seed,
)


GRAPH_FILE = "graphs/web_ids23_knn_graph.pt"
RESULTS_DIR = "results_knn"
MODEL_NAME = "graphsage"
DEFAULT_SEEDS = "42,7,13,21,100"

HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.3
LR = 5e-3
WEIGHT_DECAY = 5e-4
EPOCHS = 200
PATIENCE = 30
THRESHOLD_PERCENTILE = 95.0


class GraphSAGE(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_dim, hidden_dim, aggr="mean"))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim, aggr="mean"))
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return self.classifier(x).squeeze(-1)


def train_one_seed(data, seed, device, threshold_percentile):
    set_seed(seed)

    y = data.y.float()
    pos = (data.train_mask & (data.y == 1)).sum().item()
    neg = (data.train_mask & (data.y == 0)).sum().item()
    pos_weight = torch.tensor([neg / max(pos, 1)], device=device)

    model = GraphSAGE(
        in_dim=data.x.shape[1],
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_auc = -np.inf
    best_state = None
    epochs_since_best = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = loss_fn(logits[data.train_mask], y[data.train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(data.x, data.edge_index)
            val_scores = torch.sigmoid(val_logits[data.val_mask]).cpu().numpy()
            val_labels_np = data.y[data.val_mask].cpu().numpy()
        val_eval = evaluate_scores(
            val_labels_np,
            val_scores,
            threshold_percentile=threshold_percentile,
        )

        if val_eval["roc_auc"] > best_val_auc + 1e-4:
            best_val_auc = val_eval["roc_auc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_since_best = 0
        else:
            epochs_since_best += 1

        if epoch % 25 == 0 or epoch == 1:
            print(
                f"  epoch {epoch:3d} | train_loss {loss.item():.4f}"
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
        logits = model(data.x, data.edge_index)
        scores = torch.sigmoid(logits).cpu().numpy()

    val_labels = data.y[data.val_mask].cpu().numpy()
    val_scores = scores[data.val_mask.cpu().numpy()]
    val_metrics = evaluate_scores(
        val_labels,
        val_scores,
        threshold_percentile=threshold_percentile,
    )

    test_labels = data.y[data.test_mask].cpu().numpy()
    test_scores = scores[data.test_mask.cpu().numpy()]
    test_metrics = evaluate_scores(
        test_labels,
        test_scores,
        threshold=val_metrics["threshold"],
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

    print(f"Loading {args.graph_file}...")
    data = torch.load(args.graph_file, weights_only=False, map_location=device)
    data = data.to(device)
    print(f"Nodes: {data.x.shape[0]:,}  Edges: {data.edge_index.shape[1]:,}")
    print(f"Node features: {data.x.shape[1]}")
    train_pos = int((data.train_mask & (data.y == 1)).sum().item())
    train_neg = int((data.train_mask & (data.y == 0)).sum().item())
    print(f"Train pos/neg: {train_pos}/{train_neg}")
    print(f"Device: {device}")
    print(f"Seeds: {seeds}")
    print(f"Results dir: {args.results_dir}")

    config = {
        "graph_file": args.graph_file,
        "model": MODEL_NAME,
        "task": "node_classification",
        "hidden_dim": HIDDEN_DIM,
        "num_layers": NUM_LAYERS,
        "dropout": DROPOUT,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "epochs": EPOCHS,
        "patience": PATIENCE,
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
