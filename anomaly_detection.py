import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from torch import nn
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.utils import coalesce


GRAPH_FILE = "graphs/web_ids23_graph.pt"
DEFAULT_SEEDS = "42,7,13,21,100"
RESULTS_DIR = "results"


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def maybe_sample(indices, max_count, rng):
    if max_count is None or max_count <= 0 or len(indices) <= max_count:
        return indices
    return rng.choice(indices, size=max_count, replace=False)


def triplet_features(data, indices):
    src = data.edge_index[0, indices]
    dst = data.edge_index[1, indices]
    return torch.cat(
        [
            data.x[src],
            data.edge_attr[indices],
            data.x[dst],
        ],
        dim=1,
    )


def threshold_from_benign_scores(labels, scores, percentile):
    benign_scores = scores[labels == 0]
    return np.percentile(benign_scores, percentile)


def metric_dict(labels, scores, threshold):
    preds = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()

    false_positive_rate = fp / (fp + tn) if fp + tn else 0.0
    true_positive_rate = tp / (tp + fn) if tp + fn else 0.0

    return {
        "roc_auc": roc_auc_score(labels, scores),
        "pr_auc": average_precision_score(labels, scores),
        "fpr": false_positive_rate,
        "attack_recall": true_positive_rate,
        "threshold": threshold,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def print_threshold_report(name, labels, scores, threshold):
    metrics = metric_dict(labels, scores, threshold)
    preds = (scores >= threshold).astype(int)
    tn = metrics["tn"]
    fp = metrics["fp"]
    fn = metrics["fn"]
    tp = metrics["tp"]

    print(f"\n{name} threshold report:")
    print(f"Threshold: {threshold:.6f}")
    print(f"False positive rate: {metrics['fpr']:.4f}")
    print(f"Attack recall / detection rate: {metrics['attack_recall']:.4f}")
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(np.array([[tn, fp], [fn, tp]]))
    print(
        classification_report(
            labels,
            preds,
            target_names=["benign", "attack"],
            digits=4,
        )
    )


def evaluate_scores(labels, scores, threshold=None, threshold_percentile=95.0):
    if threshold is None:
        threshold = threshold_from_benign_scores(
            labels,
            scores,
            threshold_percentile,
        )

    return metric_dict(labels, scores, threshold)


def print_seed_result(model_name, seed, val_metrics, test_metrics):
    print(
        f"{model_name} seed {seed}: "
        f"val ROC {val_metrics['roc_auc']:.4f}, "
        f"test ROC {test_metrics['roc_auc']:.4f}, "
        f"test PR {test_metrics['pr_auc']:.4f}, "
        f"test FPR {test_metrics['fpr']:.4f}, "
        f"test recall {test_metrics['attack_recall']:.4f}"
    )


def metrics_block(split_name, metrics):
    return "\n".join([
        f"{split_name}:",
        f"  ROC-AUC: {metrics['roc_auc']:.6f}",
        f"  PR-AUC: {metrics['pr_auc']:.6f}",
        f"  Threshold: {metrics['threshold']:.6f}",
        f"  False positive rate: {metrics['fpr']:.6f}",
        f"  Attack recall / detection rate: {metrics['attack_recall']:.6f}",
        "  Confusion matrix [[TN, FP], [FN, TP]]:",
        f"  [[{metrics['tn']}, {metrics['fp']}],",
        f"   [{metrics['fn']}, {metrics['tp']}]]",
    ])


def save_seed_result(results_dir, model_name, seed, result, config):
    model_dir = os.path.join(results_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, f"seed_{seed}.txt")

    lines = [
        f"Model: {model_name}",
        f"Seed: {seed}",
        "",
        "Config:",
        *[f"  {key}: {value}" for key, value in config.items()],
        "",
        metrics_block("Validation", result["validation"]),
        "",
        metrics_block("Test", result["test"]),
        "",
    ]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return path


def summary_text(model_name, model_results):
    keys = [
        "roc_auc",
        "pr_auc",
        "fpr",
        "attack_recall",
    ]

    lines = [
        f"Model: {model_name}",
        f"Runs: {len(model_results)}",
        "",
        "Mean +/- std across seeds",
    ]

    for split in ["validation", "test"]:
        lines.append("")
        lines.append(f"{split.title()}:")
        for key in keys:
            values = np.array([
                seed_result[split][key]
                for seed_result in model_results
            ])
            std = values.std(ddof=1) if len(values) > 1 else 0.0
            lines.append(f"  {key}: {values.mean():.6f} +/- {std:.6f}")

    return "\n".join(lines) + "\n"


def save_summary(results_dir, model_name, model_results):
    model_dir = os.path.join(results_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, "summary.txt")

    with open(path, "w", encoding="utf-8") as f:
        f.write(summary_text(model_name, model_results))

    return path


def summarize_results(results, results_dir):
    print("\n=== Mean +/- std across seeds ===")

    for model_name, model_results in results.items():
        summary_path = save_summary(results_dir, model_name, model_results)
        print(f"\n{model_name}")
        print(summary_text(model_name, model_results), end="")
        print(f"Saved summary: {summary_path}")


def run_isolation_forest(data, train_idx, val_idx, test_idx, threshold_percentile, seed):
    print("\n=== Isolation Forest on full triplet features ===")

    x_train = triplet_features(data, train_idx).cpu().numpy()
    x_val = triplet_features(data, val_idx).cpu().numpy()
    x_test = triplet_features(data, test_idx).cpu().numpy()

    model = IsolationForest(
        n_estimators=300,
        contamination="auto",
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(x_train)

    val_scores = -model.score_samples(x_val)
    test_scores = -model.score_samples(x_test)

    val_labels = data.edge_label[val_idx].cpu().numpy()
    test_labels = data.edge_label[test_idx].cpu().numpy()
    val_metrics = evaluate_scores(
        val_labels,
        val_scores,
        threshold_percentile=threshold_percentile,
    )
    test_metrics = evaluate_scores(
        test_labels,
        test_scores,
        threshold=val_metrics["threshold"],
    )

    return {
        "validation": val_metrics,
        "test": test_metrics,
    }


class FeatureAutoencoder(nn.Module):
    def __init__(self, in_channels, hidden_channels=256, latent_channels=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, latent_channels),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, in_channels),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def batched_feature_scores(model, x, batch_size):
    model.eval()
    scores = []
    with torch.no_grad():
        for start in range(0, x.size(0), batch_size):
            batch = x[start:start + batch_size]
            recon = model(batch)
            score = F.mse_loss(recon, batch, reduction="none").mean(dim=1)
            scores.append(score.cpu())
    return torch.cat(scores).numpy()


def run_feature_autoencoder(
    data,
    train_idx,
    val_idx,
    test_idx,
    epochs,
    batch_size,
    lr,
    device,
    threshold_percentile,
    seed,
):
    print("\n=== Feature Autoencoder on full triplet features ===")
    set_seed(seed)

    x_train = triplet_features(data, train_idx).to(device)
    x_val = triplet_features(data, val_idx).to(device)
    x_test = triplet_features(data, test_idx).to(device)

    model = FeatureAutoencoder(x_train.size(1)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(x_train.size(0), device=device)
        total_loss = 0.0

        for start in range(0, x_train.size(0), batch_size):
            idx = perm[start:start + batch_size]
            batch = x_train[idx]

            optimizer.zero_grad()
            recon = model(batch)
            loss = F.mse_loss(recon, batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch.size(0)

        if epoch == 1 or epoch % 5 == 0:
            print(f"Epoch {epoch:03d} loss: {total_loss / x_train.size(0):.6f}")

    val_scores = batched_feature_scores(model, x_val, batch_size)
    test_scores = batched_feature_scores(model, x_test, batch_size)

    val_labels = data.edge_label[val_idx].cpu().numpy()
    test_labels = data.edge_label[test_idx].cpu().numpy()
    val_metrics = evaluate_scores(
        val_labels,
        val_scores,
        threshold_percentile=threshold_percentile,
    )
    test_metrics = evaluate_scores(
        test_labels,
        test_scores,
        threshold=val_metrics["threshold"],
    )

    return {
        "validation": val_metrics,
        "test": test_metrics,
    }


class GCNNodeEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)


class GraphEdgeFeatureAutoencoder(nn.Module):
    def __init__(
        self,
        node_in_channels,
        edge_in_channels,
        hidden_channels=128,
        node_latent_channels=64,
    ):
        super().__init__()
        self.node_encoder = GCNNodeEncoder(
            node_in_channels,
            hidden_channels,
            node_latent_channels,
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear((2 * node_latent_channels) + edge_in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
        )
        self.edge_decoder = nn.Linear(hidden_channels, edge_in_channels)

    def forward(self, x, graph_edge_index, flow_edge_index, edge_attr):
        z = self.node_encoder(x, graph_edge_index)
        src = flow_edge_index[0]
        dst = flow_edge_index[1]
        edge_input = torch.cat([z[src], edge_attr, z[dst]], dim=1)
        h = self.edge_encoder(edge_input)
        return self.edge_decoder(h)


def batched_graph_scores(
    model,
    x,
    graph_edge_index,
    flow_edge_index,
    edge_attr,
    batch_size,
):
    model.eval()
    scores = []
    with torch.no_grad():
        z = model.node_encoder(x, graph_edge_index)
        for start in range(0, edge_attr.size(0), batch_size):
            end = start + batch_size
            edges = flow_edge_index[:, start:end]
            attrs = edge_attr[start:end]
            src = edges[0]
            dst = edges[1]
            edge_input = torch.cat([z[src], attrs, z[dst]], dim=1)
            h = model.edge_encoder(edge_input)
            recon = model.edge_decoder(h)
            score = F.mse_loss(recon, attrs, reduction="none").mean(dim=1)
            scores.append(score.cpu())
    return torch.cat(scores).numpy()


class GATNodeEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, concat=True, dropout=0.0)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=0.0)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)


class GraphEdgeGATAutoencoder(nn.Module):
    def __init__(
        self,
        node_in_channels,
        edge_in_channels,
        hidden_channels=128,
        node_latent_channels=64,
        heads=4,
    ):
        super().__init__()
        self.node_encoder = GATNodeEncoder(
            node_in_channels,
            hidden_channels,
            node_latent_channels,
            heads=heads,
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear((2 * node_latent_channels) + edge_in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
        )
        self.edge_decoder = nn.Linear(hidden_channels, edge_in_channels)

    def forward(self, x, graph_edge_index, flow_edge_index, edge_attr):
        z = self.node_encoder(x, graph_edge_index)
        src = flow_edge_index[0]
        dst = flow_edge_index[1]
        edge_input = torch.cat([z[src], edge_attr, z[dst]], dim=1)
        h = self.edge_encoder(edge_input)
        return self.edge_decoder(h)


def batched_gat_scores(
    model,
    x,
    graph_edge_index,
    flow_edge_index,
    edge_attr,
    batch_size,
):
    model.eval()
    scores = []
    with torch.no_grad():
        z = model.node_encoder(x, graph_edge_index)
        for start in range(0, edge_attr.size(0), batch_size):
            end = start + batch_size
            edges = flow_edge_index[:, start:end]
            attrs = edge_attr[start:end]
            src = edges[0]
            dst = edges[1]
            edge_input = torch.cat([z[src], attrs, z[dst]], dim=1)
            h = model.edge_encoder(edge_input)
            recon = model.edge_decoder(h)
            score = F.mse_loss(recon, attrs, reduction="none").mean(dim=1)
            scores.append(score.cpu())
    return torch.cat(scores).numpy()


def run_gat_edge_autoencoder(
    data,
    train_idx,
    val_idx,
    test_idx,
    epochs,
    batch_size,
    lr,
    hidden_channels,
    latent_channels,
    device,
    threshold_percentile,
    seed,
    heads=4,
):
    print("\n=== GAT edge-feature autoencoder ===")
    print("Uses node features + graph attention message passing + edge flow features.")
    set_seed(seed)

    x = data.x.to(device)
    train_flow_edges = data.edge_index[:, train_idx].to(device)
    train_edge_attr = data.edge_attr[train_idx].to(device)

    graph_edge_index = coalesce(train_flow_edges)

    model = GraphEdgeGATAutoencoder(
        node_in_channels=data.x.size(1),
        edge_in_channels=data.edge_attr.size(1),
        hidden_channels=hidden_channels,
        node_latent_channels=latent_channels,
        heads=heads,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(train_edge_attr.size(0), device=device)
        total_loss = 0.0

        for start in range(0, train_edge_attr.size(0), batch_size):
            idx = perm[start:start + batch_size]
            batch_edges = train_flow_edges[:, idx]
            batch_attr = train_edge_attr[idx]

            optimizer.zero_grad()
            recon = model(x, graph_edge_index, batch_edges, batch_attr)
            loss = F.mse_loss(recon, batch_attr)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_attr.size(0)

        if epoch == 1 or epoch % 5 == 0:
            print(f"Epoch {epoch:03d} loss: {total_loss / train_edge_attr.size(0):.6f}")

    val_edges = data.edge_index[:, val_idx].to(device)
    val_attr = data.edge_attr[val_idx].to(device)
    test_edges = data.edge_index[:, test_idx].to(device)
    test_attr = data.edge_attr[test_idx].to(device)

    val_scores = batched_gat_scores(model, x, graph_edge_index, val_edges, val_attr, batch_size)
    test_scores = batched_gat_scores(model, x, graph_edge_index, test_edges, test_attr, batch_size)

    val_labels = data.edge_label[val_idx].cpu().numpy()
    test_labels = data.edge_label[test_idx].cpu().numpy()
    val_metrics = evaluate_scores(val_labels, val_scores, threshold_percentile=threshold_percentile)
    test_metrics = evaluate_scores(test_labels, test_scores, threshold=val_metrics["threshold"])

    return {
        "validation": val_metrics,
        "test": test_metrics,
    }


def run_graph_edge_autoencoder(
    data,
    train_idx,
    val_idx,
    test_idx,
    epochs,
    batch_size,
    lr,
    hidden_channels,
    latent_channels,
    device,
    threshold_percentile,
    seed,
):
    print("\n=== Graph edge-feature autoencoder ===")
    print("Uses node features + graph message passing + edge flow features.")
    set_seed(seed)

    x = data.x.to(device)
    train_flow_edges = data.edge_index[:, train_idx].to(device)
    train_edge_attr = data.edge_attr[train_idx].to(device)

    graph_edge_index = coalesce(train_flow_edges)

    model = GraphEdgeFeatureAutoencoder(
        node_in_channels=data.x.size(1),
        edge_in_channels=data.edge_attr.size(1),
        hidden_channels=hidden_channels,
        node_latent_channels=latent_channels,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(train_edge_attr.size(0), device=device)
        total_loss = 0.0

        for start in range(0, train_edge_attr.size(0), batch_size):
            idx = perm[start:start + batch_size]
            batch_edges = train_flow_edges[:, idx]
            batch_attr = train_edge_attr[idx]

            optimizer.zero_grad()
            recon = model(x, graph_edge_index, batch_edges, batch_attr)
            loss = F.mse_loss(recon, batch_attr)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_attr.size(0)

        if epoch == 1 or epoch % 5 == 0:
            print(f"Epoch {epoch:03d} loss: {total_loss / train_edge_attr.size(0):.6f}")

    val_edges = data.edge_index[:, val_idx].to(device)
    val_attr = data.edge_attr[val_idx].to(device)
    test_edges = data.edge_index[:, test_idx].to(device)
    test_attr = data.edge_attr[test_idx].to(device)

    val_scores = batched_graph_scores(
        model,
        x,
        graph_edge_index,
        val_edges,
        val_attr,
        batch_size,
    )
    test_scores = batched_graph_scores(
        model,
        x,
        graph_edge_index,
        test_edges,
        test_attr,
        batch_size,
    )

    val_labels = data.edge_label[val_idx].cpu().numpy()
    test_labels = data.edge_label[test_idx].cpu().numpy()
    val_metrics = evaluate_scores(
        val_labels,
        val_scores,
        threshold_percentile=threshold_percentile,
    )
    test_metrics = evaluate_scores(
        test_labels,
        test_scores,
        threshold=val_metrics["threshold"],
    )

    return {
        "validation": val_metrics,
        "test": test_metrics,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-file", default=GRAPH_FILE)
    parser.add_argument(
        "--model",
        choices=["all", "isoforest", "feature_ae", "graph_edge_ae", "gat"],
        default="all",
    )
    parser.add_argument("--gat-heads", type=int, default=4, help="Number of attention heads for GAT.")
    parser.add_argument(
        "--max-train",
        type=int,
        default=0,
        help="0 means use all benign training edges.",
    )
    parser.add_argument(
        "--max-eval",
        type=int,
        default=0,
        help="0 means use all validation/test edges.",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-channels", type=int, default=128)
    parser.add_argument("--latent-channels", type=int, default=64)
    parser.add_argument(
        "--threshold-percentile",
        type=float,
        default=95.0,
        help="Percentile of validation benign anomaly scores used as threshold.",
    )
    parser.add_argument(
        "--seeds",
        default=DEFAULT_SEEDS,
        help="Comma-separated seeds to run and average.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--results-dir",
        default=RESULTS_DIR,
        help="Directory where per-seed and summary result files are saved.",
    )

    args = parser.parse_args()
    seeds = [
        int(seed.strip())
        for seed in args.seeds.split(",")
        if seed.strip()
    ]

    data = torch.load(args.graph_file, weights_only=False, map_location="cpu")

    train_idx = torch.where(data.train_mask)[0].numpy()
    val_idx = torch.where(data.val_mask)[0].numpy()
    test_idx = torch.where(data.test_mask)[0].numpy()

    print("\nLoaded graph:")
    print(f"Graph file: {args.graph_file}")
    print(f"Nodes: {data.x.shape[0]:,}")
    print(f"Edges: {data.edge_index.shape[1]:,}")
    print(f"Node features: {data.x.shape[1]}")
    print(f"Edge features: {data.edge_attr.shape[1]}")
    print(f"Triplet features: {data.x.shape[1] * 2 + data.edge_attr.shape[1]}")
    print(f"Device: {args.device}")
    print(f"Seeds: {seeds}")
    print(f"Results dir: {args.results_dir}")

    print("\nFull split sizes:")
    print(f"Train: {len(train_idx):,}")
    print(f"Val: {len(val_idx):,}")
    print(f"Test: {len(test_idx):,}")

    models = (
        ["isoforest", "feature_ae", "graph_edge_ae", "gat"]
        if args.model == "all"
        else [args.model]
    )

    all_results = {
        model_name: []
        for model_name in models
    }

    config = {
        "graph_file": args.graph_file,
        "max_train": args.max_train,
        "max_eval": args.max_eval,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "hidden_channels": args.hidden_channels,
        "latent_channels": args.latent_channels,
        "threshold_percentile": args.threshold_percentile,
        "gat_heads": args.gat_heads,
        "device": args.device,
        "train_split_size": len(train_idx),
        "val_split_size": len(val_idx),
        "test_split_size": len(test_idx),
    }

    for seed in seeds:
        print(f"\n\n######## Seed {seed} ########")
        rng = np.random.default_rng(seed)
        seed_train_idx = maybe_sample(train_idx, args.max_train, rng)
        seed_val_idx = maybe_sample(val_idx, args.max_eval, rng)
        seed_test_idx = maybe_sample(test_idx, args.max_eval, rng)

        print("\nUsing samples:")
        print(f"Train: {len(seed_train_idx):,}")
        print(f"Val: {len(seed_val_idx):,}")
        print(f"Test: {len(seed_test_idx):,}")

        if "isoforest" in models:
            result = run_isolation_forest(
                data,
                seed_train_idx,
                seed_val_idx,
                seed_test_idx,
                args.threshold_percentile,
                seed,
            )
            all_results["isoforest"].append(result)
            path = save_seed_result(
                args.results_dir,
                "isoforest",
                seed,
                result,
                config,
            )
            print_seed_result("isoforest", seed, result["validation"], result["test"])
            print(f"Saved run: {path}")

        if "feature_ae" in models:
            result = run_feature_autoencoder(
                data,
                seed_train_idx,
                seed_val_idx,
                seed_test_idx,
                args.epochs,
                args.batch_size,
                args.lr,
                args.device,
                args.threshold_percentile,
                seed,
            )
            all_results["feature_ae"].append(result)
            path = save_seed_result(
                args.results_dir,
                "feature_ae",
                seed,
                result,
                config,
            )
            print_seed_result("feature_ae", seed, result["validation"], result["test"])
            print(f"Saved run: {path}")

        if "graph_edge_ae" in models:
            result = run_graph_edge_autoencoder(
                data,
                seed_train_idx,
                seed_val_idx,
                seed_test_idx,
                args.epochs,
                args.batch_size,
                args.lr,
                args.hidden_channels,
                args.latent_channels,
                args.device,
                args.threshold_percentile,
                seed,
            )
            all_results["graph_edge_ae"].append(result)
            path = save_seed_result(
                args.results_dir,
                "graph_edge_ae",
                seed,
                result,
                config,
            )
            print_seed_result(
                "graph_edge_ae",
                seed,
                result["validation"],
                result["test"],
            )
            print(f"Saved run: {path}")

        if "gat" in models:
            result = run_gat_edge_autoencoder(
                data,
                seed_train_idx,
                seed_val_idx,
                seed_test_idx,
                args.epochs,
                args.batch_size,
                args.lr,
                args.hidden_channels,
                args.latent_channels,
                args.device,
                args.threshold_percentile,
                seed,
                heads=args.gat_heads,
            )
            all_results["gat"].append(result)
            path = save_seed_result(
                args.results_dir,
                "gat",
                seed,
                result,
                config,
            )
            print_seed_result("gat", seed, result["validation"], result["test"])
            print(f"Saved run: {path}")

    summarize_results(all_results, args.results_dir)


if __name__ == "__main__":
    main()
