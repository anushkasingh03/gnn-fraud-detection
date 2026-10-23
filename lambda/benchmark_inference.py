"""
Benchmark local ONNX inference on real Elliptic transactions.

Measures:
- PyTorch vs ONNX prediction agreement
- P50, P95, and P99 ONNX inference latency
- P50, P95, and P99 total handler latency
- Local subgraph sizes
- Prediction agreement at threshold 0.5

Run from the project root:

    MODEL_PATH=models/real_baseline_sage.onnx \
    python lambda/benchmark_inference.py
"""

import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "src")
sys.path.insert(0, "lambda")

from data_loader import load_elliptic, time_based_split
from models import GraphSAGE
from lambda_handler import handler, _get_session


N_SAMPLES = 100
SEED = 42
THRESHOLD = 0.5

CHECKPOINT = "models/real_baseline_sage.pt"
DATA_DIR = "data/elliptic_real"
OUTPUT_PATH = "outputs/real_onnx_benchmark.json"


def extract_subgraph(data, target, num_hops=3):
    """
    Extract the incoming k-hop neighborhood for a target node.

    The target is always local node 0.
    """

    src, dst = data.edge_index

    selected = torch.tensor(
        [target],
        dtype=torch.long,
    )

    for _ in range(num_hops):
        incoming = torch.isin(dst, selected)

        selected = torch.unique(
            torch.cat([selected, src[incoming]])
        )

    neighbors = selected[selected != target]

    nodes = torch.cat([
        torch.tensor([target]),
        neighbors,
    ])

    global_to_local = torch.full(
        (data.x.shape[0],),
        -1,
        dtype=torch.long,
    )

    global_to_local[nodes] = torch.arange(len(nodes))

    keep_edges = (
        (global_to_local[src] >= 0)
        & (global_to_local[dst] >= 0)
    )

    local_edges = torch.stack([
        global_to_local[src[keep_edges]],
        global_to_local[dst[keep_edges]],
    ])

    return nodes, local_edges


def main():

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("Loading Elliptic dataset...")

    data, _ = load_elliptic(DATA_DIR)

    _, _, test_mask = time_based_split(data)

    print("Loading PyTorch GraphSAGE...")

    model = GraphSAGE(
        in_channels=data.x.shape[1],
        hidden_channels=128,
    )

    state = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(state)
    model.eval()

    # Full-graph predictions are the reference.
    with torch.no_grad():
        full_logits = model(
            data.x,
            data.edge_index,
        )

        full_probs = torch.sigmoid(
            full_logits
        ).cpu().numpy()

    # Select both predicted fraudulent and legitimate transactions.
    test_indices = torch.where(test_mask)[0].numpy()

    predicted_fraud = test_indices[
        full_probs[test_indices] >= THRESHOLD
    ]

    predicted_legit = test_indices[
        full_probs[test_indices] < THRESHOLD
    ]

    rng = np.random.default_rng(SEED)

    n_fraud = min(
        N_SAMPLES // 2,
        len(predicted_fraud),
    )

    n_legit = min(
        N_SAMPLES - n_fraud,
        len(predicted_legit),
    )

    selected_fraud = rng.choice(
        predicted_fraud,
        size=n_fraud,
        replace=False,
    )

    selected_legit = rng.choice(
        predicted_legit,
        size=n_legit,
        replace=False,
    )

    selected = np.concatenate([
        selected_fraud,
        selected_legit,
    ])

    rng.shuffle(selected)

    print(f"Benchmarking {len(selected)} transactions...")
    print(f"Predicted fraudulent: {n_fraud}")
    print(f"Predicted legitimate: {n_legit}")

    # Warm up ONNX Runtime before recording latency.
    session = _get_session()

    warmup_x = np.zeros(
        (2, data.x.shape[1]),
        dtype=np.float32,
    )

    warmup_edges = np.array(
        [[0, 1], [1, 0]],
        dtype=np.int64,
    )

    for _ in range(10):
        session.run(
            None,
            {
                "x": warmup_x,
                "edge_index": warmup_edges,
            },
        )

    inference_times = []
    handler_times = []
    neighborhood_sizes = []

    absolute_differences = []
    prediction_agreements = []

    results = []

    start = time.perf_counter()

    for i, target in enumerate(selected, start=1):

        target = int(target)

        nodes, local_edges = extract_subgraph(
            data,
            target,
            num_hops=3,
        )

        event = {
            "node_features": (
                data.x[nodes].numpy().tolist()
            ),
            "edge_index": (
                local_edges.numpy().tolist()
            ),
            "threshold": THRESHOLD,
        }

        response = handler(event, None)

        if response["statusCode"] != 200:
            raise RuntimeError(
                f"Handler failed for node {target}: "
                f"{response['body']}"
            )

        body = json.loads(response["body"])

        onnx_prob = body["fraud_probability"]
        torch_prob = float(full_probs[target])

        difference = abs(
            onnx_prob - torch_prob
        )

        agreement = (
            (onnx_prob >= THRESHOLD)
            == (torch_prob >= THRESHOLD)
        )

        inference_times.append(
            body["inference_ms"]
        )

        handler_times.append(
            body["total_handler_ms"]
        )

        neighborhood_sizes.append(
            body["num_nodes_in_subgraph"]
        )

        absolute_differences.append(difference)
        prediction_agreements.append(agreement)

        results.append({
            "node_id": target,
            "label": int(data.y[target].item()),
            "pytorch_probability": torch_prob,
            "onnx_probability": onnx_prob,
            "absolute_difference": difference,
            "prediction_agreement": agreement,
            "inference_ms": body["inference_ms"],
            "handler_ms": body["total_handler_ms"],
            "subgraph_nodes": body["num_nodes_in_subgraph"],
        })

        if i % 20 == 0:
            print(
                f"Completed {i}/{len(selected)}"
            )

    elapsed = time.perf_counter() - start

    # Summarize latency.
    def latency_summary(values):
        return {
            "mean_ms": float(np.mean(values)),
            "p50_ms": float(np.percentile(values, 50)),
            "p95_ms": float(np.percentile(values, 95)),
            "p99_ms": float(np.percentile(values, 99)),
            "max_ms": float(np.max(values)),
        }

    summary = {
        "n_transactions": len(selected),
        "n_predicted_fraud": n_fraud,
        "n_predicted_legitimate": n_legit,
        "inference_latency": latency_summary(inference_times),
        "handler_latency": latency_summary(handler_times),
        "subgraph_nodes": {
            "min": int(np.min(neighborhood_sizes)),
            "median": float(np.median(neighborhood_sizes)),
            "max": int(np.max(neighborhood_sizes)),
        },
        "prediction_agreement_rate": float(
            np.mean(prediction_agreements)
        ),
        "max_absolute_probability_difference": float(
            np.max(absolute_differences)
        ),
        "total_benchmark_seconds": elapsed,
    }

    os.makedirs("outputs", exist_ok=True)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(
            {
                "summary": summary,
                "transactions": results,
            },
            f,
            indent=2,
        )

    print("\n" + "=" * 55)
    print("ONNX INFERENCE BENCHMARK")
    print("=" * 55)

    print(
        f"\nTransactions tested: "
        f"{summary['n_transactions']}"
    )

    print(
        f"Prediction agreement: "
        f"{summary['prediction_agreement_rate'] * 100:.2f}%"
    )

    print(
        f"Maximum probability difference: "
        f"{summary['max_absolute_probability_difference']:.8f}"
    )

    print("\nONNX inference latency (ms):")
    print(json.dumps(
        summary["inference_latency"],
        indent=2,
    ))

    print("\nTotal handler latency (ms):")
    print(json.dumps(
        summary["handler_latency"],
        indent=2,
    ))

    print("\nSubgraph sizes:")
    print(json.dumps(
        summary["subgraph_nodes"],
        indent=2,
    ))

    print(
        f"\nResults saved to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()