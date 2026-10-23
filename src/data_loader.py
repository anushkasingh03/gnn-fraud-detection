"""
Load the Elliptic Bitcoin dataset into a PyTorch Geometric graph.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


def load_elliptic(data_dir: str):
    data_dir = Path(data_dir)

    feat_path = data_dir / "elliptic_txs_features.csv"
    class_path = data_dir / "elliptic_txs_classes.csv"
    edge_path = data_dir / "elliptic_txs_edgelist.csv"

    for path in (feat_path, class_path, edge_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing dataset file: {path}")

    # Load CSV files
    feat_df = pd.read_csv(feat_path, header=None)
    class_df = pd.read_csv(class_path)
    edge_df = pd.read_csv(edge_path)

    if feat_df.shape[1] < 3:
        raise ValueError("Features CSV must contain transaction ID, time step, and features")

    feat_df.columns = (
        ["txId", "time_step"]
        + [f"f{i}" for i in range(feat_df.shape[1] - 2)]
    )

    class_df.columns = [str(c).strip() for c in class_df.columns]

    if "txId" not in class_df.columns:
        class_df = class_df.rename(columns={class_df.columns[0]: "txId"})

    # Validate transaction IDs
    if feat_df["txId"].isna().any():
        raise ValueError("Missing transaction IDs")

    if feat_df["txId"].duplicated().any():
        raise ValueError("Duplicate transaction IDs")

    id_to_idx = {
        tx_id: idx
        for idx, tx_id in enumerate(feat_df["txId"].values)
    }

    # Map labels: illicit=1, licit=0, unknown=-1
    label_map = {"1": 1, "2": 0, "unknown": -1}

    labels = class_df["class"].astype(str).str.strip()

    invalid_labels = set(labels.unique()) - set(label_map)

    if invalid_labels:
        raise ValueError(f"Unexpected labels: {invalid_labels}")

    class_df["label"] = labels.map(label_map)
    class_df["node_idx"] = class_df["txId"].map(id_to_idx)

    class_df = class_df.dropna(subset=["node_idx"])
    class_df["node_idx"] = class_df["node_idx"].astype(int)

    y = np.full(len(feat_df), -1, dtype=np.int64)
    y[class_df["node_idx"].values] = class_df["label"].values

    # Load numerical features
    x = feat_df.drop(columns=["txId", "time_step"]).to_numpy(
        dtype=np.float32
    )

    if not np.isfinite(x).all():
        raise ValueError("Features contain NaN or infinity")

    time_step = feat_df["time_step"].to_numpy(dtype=np.int64)

    # Convert transaction edges into graph indices
    src = edge_df.iloc[:, 0].map(id_to_idx)
    dst = edge_df.iloc[:, 1].map(id_to_idx)

    valid = src.notna() & dst.notna()

    edge_index = np.vstack([
        src[valid].astype(np.int64).values,
        dst[valid].astype(np.int64).values,
    ])

    # Construct PyTorch Geometric graph
    data = Data(
        x=torch.tensor(x, dtype=torch.float32),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        y=torch.tensor(y, dtype=torch.long),
    )

    data.time_step = torch.tensor(time_step, dtype=torch.long)

    # Check whether edges cross time-step boundaries
    src_idx, dst_idx = data.edge_index

    cross_time_edges = int(
        (data.time_step[src_idx] != data.time_step[dst_idx])
        .sum()
        .item()
    )

    print(f"Cross-time-step edges: {cross_time_edges}")

    labeled = data.y >= 0

    meta = {
        "n_nodes": data.num_nodes,
        "n_edges": data.edge_index.shape[1],
        "n_features": data.x.shape[1],
        "n_labeled": int(labeled.sum()),
        "n_illicit": int((data.y == 1).sum()),
        "n_licit": int((data.y == 0).sum()),
        "illicit_rate_labeled": (
            float((data.y == 1).sum()) / max(int(labeled.sum()), 1)
        ),
        "cross_time_edges": cross_time_edges,
    }

    return data, meta


def time_based_split(data, train_max_ts=34, val_max_ts=39):
    """Split labeled transactions chronologically."""

    if train_max_ts >= val_max_ts:
        raise ValueError("train_max_ts must be smaller than val_max_ts")

    labeled = data.y >= 0
    ts = data.time_step

    train_mask = labeled & (ts <= train_max_ts)

    val_mask = (
        labeled
        & (ts > train_max_ts)
        & (ts <= val_max_ts)
    )

    test_mask = labeled & (ts > val_max_ts)

    return train_mask, val_mask, test_mask


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default="data/elliptic_synthetic",
    )
    args = parser.parse_args()

    data, meta = load_elliptic(args.data_dir)

    print("\nDataset metadata:")
    for key, value in meta.items():
        print(f"{key}: {value}")

    train_mask, val_mask, test_mask = time_based_split(data)

    print("\nTemporal split:")
    print(f"Train: {train_mask.sum().item()}")
    print(f"Validation: {val_mask.sum().item()}")
    print(f"Test: {test_mask.sum().item()}")