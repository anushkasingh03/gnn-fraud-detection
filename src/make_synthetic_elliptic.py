"""
Generates a fake dataset with the same shape and file format as Elliptic.

Handy for testing the pipeline end to end without downloading the real data
(e.g. in CI or on a machine without a Kaggle account). The numbers you get on
this data are meaningless; use the real dataset for actual results.

    python3 src/make_synthetic_elliptic.py   # writes to data/elliptic_synthetic/
"""
import os

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

N_NODES      = 203_769
N_TIME_STEPS = 49
N_FEATURES   = 165          # + txId + time_step = 167 columns, same as the real file
ILLICIT_FRAC = 4_545 / N_NODES
LICIT_FRAC   = 42_019 / N_NODES
AVG_DEGREE   = 234_355 / N_NODES

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "elliptic_synthetic")


def make_features():
    time_steps = rng.integers(1, N_TIME_STEPS + 1, size=N_NODES)
    u = rng.random(N_NODES)
    classes = np.where(u < ILLICIT_FRAC, 1, np.where(u < ILLICIT_FRAC + LICIT_FRAC, 2, 3))

    x = rng.normal(0.0, 1.0, (N_NODES, N_FEATURES))
    # nudge illicit nodes so there's actually something to learn
    illicit = classes == 1
    shift = rng.normal(0.6, 0.3, N_FEATURES)
    noise = rng.normal(0.0, 1.4, (illicit.sum(), N_FEATURES))
    x[illicit] = x[illicit] * 0.5 + shift + noise * 0.5

    tx_ids = np.arange(1, N_NODES + 1)
    feat_df = pd.DataFrame(x)
    feat_df.insert(0, "time_step", time_steps)
    feat_df.insert(0, "txId", tx_ids)

    class_df = pd.DataFrame({
        "txId": tx_ids,
        "class": np.where(classes == 1, "1", np.where(classes == 2, "2", "unknown")),
    })
    return feat_df, class_df, time_steps


def make_edges(time_steps):
    # like the real data, edges only connect transactions within the same time step
    node_ids = np.arange(1, N_NODES + 1)
    per_ts = int(AVG_DEGREE * N_NODES) // N_TIME_STEPS
    src_all, dst_all = [], []
    for ts in range(1, N_TIME_STEPS + 1):
        nodes = node_ids[time_steps == ts]
        if len(nodes) < 2:
            continue
        src = rng.choice(nodes, size=per_ts)
        dst = rng.choice(nodes, size=per_ts)
        keep = src != dst
        src_all.append(src[keep])
        dst_all.append(dst[keep])
    return (pd.DataFrame({"txId1": np.concatenate(src_all), "txId2": np.concatenate(dst_all)})
              .drop_duplicates().reset_index(drop=True))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    feat_df, class_df, time_steps = make_features()
    edge_df = make_edges(time_steps)

    feat_df.to_csv(os.path.join(OUT_DIR, "elliptic_txs_features.csv"), index=False, header=False)
    class_df.to_csv(os.path.join(OUT_DIR, "elliptic_txs_classes.csv"), index=False)
    edge_df.to_csv(os.path.join(OUT_DIR, "elliptic_txs_edgelist.csv"), index=False)

    print(f"wrote {len(feat_df)} nodes, {len(edge_df)} edges to {os.path.abspath(OUT_DIR)}")
    print(class_df["class"].value_counts().to_string())


if __name__ == "__main__":
    main()
