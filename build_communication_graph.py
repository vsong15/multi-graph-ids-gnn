import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data

INPUT_FILE = "web_ids23_merged_clean.csv"
OUTPUT_FILE = "graphs/web_ids23_communication_graph_data.pt"

df = pd.read_csv(INPUT_FILE, low_memory=False)

grouped = df.groupby(["id.orig_h", "id.resp_h"]).agg({
    "flow_duration": "mean",
    "fwd_pkts_tot": "sum",
    "bwd_pkts_tot": "sum",
    "fwd_data_pkts_tot": "sum",
    "bwd_data_pkts_tot": "sum",
    "flow_pkts_per_sec": "mean",
    "payload_bytes_per_second": "mean",
    "down_up_ratio": "mean",
    "flow_SYN_flag_count": "sum",
    "flow_ACK_flag_count": "sum",
    "flow_RST_flag_count": "sum",
    "attack": "max"
}).reset_index()

grouped["attack"] = pd.to_numeric(grouped["attack"], errors="coerce").fillna(0).astype(int)

all_nodes = pd.concat([grouped["id.orig_h"], grouped["id.resp_h"]]).unique()
node_to_idx = {node: i for i, node in enumerate(all_nodes)}

src = grouped["id.orig_h"].map(node_to_idx).values
dst = grouped["id.resp_h"].map(node_to_idx).values

edge_index = torch.from_numpy(np.vstack([src, dst]).astype(np.int64))

edge_features = grouped[[
    "flow_duration",
    "fwd_pkts_tot",
    "bwd_pkts_tot",
    "fwd_data_pkts_tot",
    "bwd_data_pkts_tot",
    "flow_pkts_per_sec",
    "payload_bytes_per_second",
    "down_up_ratio",
    "flow_SYN_flag_count",
    "flow_ACK_flag_count",
    "flow_RST_flag_count"
]].values

edge_attr = torch.tensor(edge_features, dtype=torch.float)

edge_labels = torch.tensor(grouped["attack"].values, dtype=torch.float)

src_stats = df.groupby("id.orig_h").agg({
    "fwd_pkts_tot": "sum",
    "bwd_pkts_tot": "sum",
    "flow_duration": "mean"
})

dst_stats = df.groupby("id.resp_h").agg({
    "fwd_pkts_tot": "sum",
    "bwd_pkts_tot": "sum",
    "flow_duration": "mean"
})

node_features_df = src_stats.add(dst_stats, fill_value=0)

node_features = []
for node in all_nodes:
    if node in node_features_df.index:
        node_features.append(node_features_df.loc[node].values)
    else:
        node_features.append([0, 0, 0])

x = torch.tensor(np.array(node_features), dtype=torch.float)

num_edges = edge_index.shape[1]
indices = np.arange(num_edges)

train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
train_idx, val_idx = train_test_split(train_idx, test_size=0.1, random_state=42)

train_mask = torch.zeros(num_edges, dtype=torch.bool)
val_mask = torch.zeros(num_edges, dtype=torch.bool)
test_mask = torch.zeros(num_edges, dtype=torch.bool)

train_mask[train_idx] = True
val_mask[val_idx] = True
test_mask[test_idx] = True

data = Data(
    x=x,
    edge_index=edge_index,
    edge_attr=edge_attr,
    edge_label=edge_labels
)

data.train_mask = train_mask
data.val_mask = val_mask
data.test_mask = test_mask

torch.save(data, OUTPUT_FILE)