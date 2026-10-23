"""
Export a trained GraphSAGE model to ONNX.

Run from the project root:

    python lambda/export_to_onnx.py \
        --ckpt models/real_baseline_sage.pt \
        --out models/real_baseline_sage.onnx

The exported model accepts:
    x: Node feature matrix [num_nodes, 165]
    edge_index: Graph connectivity [2, num_edges]

The model produces:
    logits: One prediction score per node.

Note:
    ONNX export compatibility must be tested with ONNX Runtime.
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from models import GraphSAGE  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",   default="models/real_baseline_sage.pt")
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--out",    default="model.onnx")
    args = parser.parse_args()

    state = torch.load(
    args.ckpt,
    map_location="cpu",
    weights_only=True,
    )
    # read input size from the checkpoint so it works for 165 (real) or other feature counts
    in_channels = state["convs.0.lin_l.weight"].shape[1]

    model = GraphSAGE(in_channels=in_channels, hidden_channels=args.hidden)
    model.load_state_dict(state)
    model.eval()

    dummy_x  = torch.randn(50, in_channels)
    dummy_ei = torch.randint(0, 50, (2, 150))

    torch.onnx.export(
        model,
        (dummy_x, dummy_ei),
        args.out,
        input_names=["x", "edge_index"],
        output_names=["logits"],
        dynamic_axes={
            "x":          {0: "num_nodes"},
            "edge_index": {1: "num_edges"},
            "logits":     {0: "num_nodes"},
        },
        opset_version=18,
    )
    output_dir = os.path.dirname(args.out)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    print(f"exported {args.out} ({os.path.getsize(args.out) / 1024:.0f} KB, {in_channels} input features)")


if __name__ == "__main__":
    main()
