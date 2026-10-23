"""
Train the GraphSAGE baseline on Elliptic.

Full-batch training (the whole graph fits on a T4 easily). Class imbalance is
handled with a weighted BCE loss; the checkpoint with the best validation F1 on
the illicit class is kept and evaluated on the held-out test time steps.

    cd src
    python3 train_baseline.py --data_dir ../data/elliptic_real --epochs 150
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn.functional as F

from data_loader import load_elliptic, time_based_split
from metrics import binary_metrics
from models import GraphSAGE


@torch.no_grad()
def evaluate(model, data, mask):
    model.eval()
    probs = torch.sigmoid(model(data.x, data.edge_index))[mask].cpu().numpy()
    return binary_metrics(probs, data.y[mask].cpu().numpy())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/elliptic_real")
    parser.add_argument("--epochs",      type=int,   default=150)
    parser.add_argument("--hidden",      type=int,   default=128)
    parser.add_argument("--lr",          type=float, default=0.005)
    parser.add_argument("--wd",          type=float, default=5e-4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--out", default="models/real_baseline_sage.pt")
    parser.add_argument("--metrics_out", default="outputs/real_baseline_metrics.json")
    args = parser.parse_args()

    #torch.manual_seed(args.seed)
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print(f"device: {device}")

    data, meta = load_elliptic(args.data_dir)
    print("dataset:", meta)
    train_mask, val_mask, test_mask = time_based_split(data)
    for name, mask in [
        ("train", train_mask),
        ("validation", val_mask),
        ("test", test_mask),
    ]:
        if not mask.any().item():
            raise ValueError(f"{name} split contains no labeled transactions")
    data = data.to(device)
    train_mask, val_mask, test_mask = train_mask.to(device), val_mask.to(device), test_mask.to(device)

    model = GraphSAGE(in_channels=data.x.shape[1], hidden_channels=args.hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # licit outnumbers illicit roughly 9:1 among labeled train nodes,
    # so upweight the positive class by that ratio
    n_pos = int((data.y[train_mask] == 1).sum())
    n_neg = int((data.y[train_mask] == 0).sum())
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)
    print(f"pos_weight: {pos_weight.item():.2f}  (illicit={n_pos}, licit={n_neg})")

    best_val_f1, best_state = -1.0, None
    history = []
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = F.binary_cross_entropy_with_logits(
            logits[train_mask], data.y[train_mask].float(), pos_weight=pos_weight
        )
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch == args.epochs:
            vm = evaluate(model, data, val_mask)
            history.append({"epoch": epoch, "loss": loss.item(), **vm})
            print(f"epoch {epoch:3d} | loss {loss.item():.4f} | "
                  f"val F1 {vm['f1_illicit']:.4f} | recall {vm['recall_illicit']:.4f} | "
                  f"FPR {vm['false_positive_rate']:.4f}")
            if vm["f1_illicit"] > best_val_f1:
                best_val_f1 = vm["f1_illicit"]
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    train_time = time.time() - t0
    model.load_state_dict(best_state)
    test_metrics = evaluate(model, data, test_mask)

    print("\ntest set (best val checkpoint):")
    print(json.dumps(test_metrics, indent=2))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    os.makedirs(os.path.dirname(args.metrics_out), exist_ok=True)
    torch.save(best_state, args.out)
    with open(args.metrics_out, "w") as f:
        json.dump({"dataset_meta": meta, "train_time_sec": train_time,
                   "test_metrics": test_metrics, "history": history}, f, indent=2)
    print(f"saved {args.out} and {args.metrics_out}")


if __name__ == "__main__":
    main()
