"""
AWS Lambda handler for real-time transaction scoring.

The request carries a small local subgraph: the transaction being scored plus
its k-hop neighbors (fetched upstream from whatever graph store holds the
transaction network). Node 0 is always the target. Running the GNN on this
subgraph instead of the full 200K-node graph is what keeps latency in the
low milliseconds.

The ONNX session is created once per execution environment and reused across
warm invocations.

Example body:
    {
      "node_features": [[...], [...], ...],   # one row per node, row 0 = target
      "edge_index": [[src...], [dst...]],
      "threshold": 0.5                         # optional
    }
"""
import json
import os
import time

import numpy as np
import onnxruntime as ort

MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    "/var/task/models/real_baseline_sage.onnx"
)
_session = None


def _get_session():
    global _session
    if _session is None:
        _session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    return _session


def handler(event, context):
    t_start = time.perf_counter()
    try:
        body = json.loads(event["body"]) if isinstance(event.get("body"), str) else event

        x = np.asarray(body["node_features"], dtype=np.float32)
        edge_index = np.asarray(body["edge_index"], dtype=np.int64)
        threshold = float(body.get("threshold", 0.5))

        sess = _get_session()
        expected = sess.get_inputs()[0].shape[1]

        if x.ndim != 2 or x.shape[1] != expected:
            raise ValueError(f"node_features must be [num_nodes, {expected}], got {list(x.shape)}")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError(f"edge_index must be [2, num_edges], got {list(edge_index.shape)}")
        if x.shape[0] == 0:
            raise ValueError("node_features must contain at least one node")

        if not np.isfinite(x).all():
            raise ValueError("node_features contains NaN or infinite values")

        if not np.isfinite(threshold) or not (0.0 <= threshold <= 1.0):
            raise ValueError("threshold must be between 0 and 1")

        if edge_index.size > 0:
            if edge_index.min() < 0 or edge_index.max() >= x.shape[0]:
                raise ValueError(
                    "edge_index contains node indices outside the subgraph"
                )

        t_infer = time.perf_counter()
        logits = sess.run(None, {"x": x, "edge_index": edge_index})[0]
        infer_ms = (time.perf_counter() - t_infer) * 1000

        logit = float(logits[0])

        if logit >= 0:
            prob = float(1.0 / (1.0 + np.exp(-logit)))
        else:
            exp_logit = np.exp(logit)
            prob = float(exp_logit / (1.0 + exp_logit))

        return {
            "statusCode": 200,
            "body": json.dumps({
                "fraud_probability": round(prob, 6),
                "flagged": prob >= threshold,
                "threshold": threshold,
                "inference_ms": round(infer_ms, 3),
                "total_handler_ms": round((time.perf_counter() - t_start) * 1000, 3),
                "num_nodes_in_subgraph": int(x.shape[0]),
            }),
        }
    except (KeyError, ValueError) as e:
        return {"statusCode": 400, "body": json.dumps({"error": str(e)})}
    except Exception as e:
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}
