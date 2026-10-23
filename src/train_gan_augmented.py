"""
GAN augmentation on top of a trained GraphSAGE baseline.

1. Load and freeze the GraphSAGE baseline.
2. Extract embeddings from the graph.
3. Train a GAN on illicit training embeddings.
4. Generate synthetic illicit embeddings.
5. Train a new classifier head using real and synthetic embeddings.
6. Select the best head using validation F1.
7. Evaluate baseline and augmented models on the same test set.

Run from the project root.
"""

import argparse
import copy
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
import torch.nn.functional as F

from data_loader import load_elliptic, time_based_split
from gan import train_gan, generate_synthetic_embeddings
from metrics import binary_metrics
from models import GraphSAGE


@torch.no_grad()
def evaluate_head(head, embeddings, labels, mask):
    head.eval()

    logits = head(embeddings[mask]).squeeze(-1)
    probs = torch.sigmoid(logits).cpu().numpy()
    targets = labels[mask].cpu().numpy()

    return binary_metrics(probs, targets)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        default="data/elliptic_real",
    )

    parser.add_argument(
        "--baseline_ckpt",
        default="models/real_baseline_sage.pt",
    )

    parser.add_argument(
        "--hidden",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--gan_epochs",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--head_epochs",
        type=int,
        default=150,
    )

    parser.add_argument(
        "--synth_mult",
        type=float,
        default=2.0,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
    )

    parser.add_argument(
        "--metrics_out",
        default="outputs/real_gan_augmented_metrics.json",
    )

    parser.add_argument(
        "--head_out",
        default="models/real_gan_head.pt",
    )

    args = parser.parse_args()

    if args.gan_epochs < 1 or args.head_epochs < 1:
        raise ValueError("Training epochs must be positive")

    if args.synth_mult < 0:
        raise ValueError("synth_mult cannot be negative")

    torch.manual_seed(args.seed)

    # Device selection
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

    # -----------------------------------------
    # 1. Load dataset and temporal splits
    # -----------------------------------------

    data, meta = load_elliptic(args.data_dir)

    train_mask, val_mask, test_mask = time_based_split(data)

    for name, mask in [
        ("train", train_mask),
        ("validation", val_mask),
        ("test", test_mask),
    ]:
        if not mask.any().item():
            raise ValueError(f"{name} split is empty")

    data = data.to(device)

    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)

    # -----------------------------------------
    # 2. Load trained GraphSAGE baseline
    # -----------------------------------------

    backbone = GraphSAGE(
        in_channels=data.x.shape[1],
        hidden_channels=args.hidden,
    ).to(device)

    backbone.load_state_dict(
        torch.load(
            args.baseline_ckpt,
            map_location=device,
            weights_only=True,
        )
    )

    backbone.eval()

    for parameter in backbone.parameters():
        parameter.requires_grad_(False)

    # -----------------------------------------
    # 3. Evaluate baseline on current dataset
    # -----------------------------------------

    with torch.no_grad():
        baseline_logits = backbone(data.x, data.edge_index)

        baseline_probs = torch.sigmoid(
            baseline_logits[test_mask]
        ).cpu().numpy()

        baseline = binary_metrics(
            baseline_probs,
            data.y[test_mask].cpu().numpy(),
        )

        all_embeds = backbone.embed(
            data.x,
            data.edge_index,
        ).detach()

    print("\nBaseline test metrics:")
    print(json.dumps(baseline, indent=2))

    # -----------------------------------------
    # 4. Extract training embeddings
    # -----------------------------------------

    real_illicit = all_embeds[
        (data.y == 1) & train_mask
    ]

    real_licit = all_embeds[
        (data.y == 0) & train_mask
    ]

    n_illicit = len(real_illicit)
    n_licit = len(real_licit)

    print(
        f"\nTraining embeddings: "
        f"illicit={n_illicit}, licit={n_licit}"
    )

    if n_illicit < 2 or n_licit < 1:
        raise ValueError("Insufficient labeled training examples")

    # -----------------------------------------
    # 5. Train GAN
    # -----------------------------------------

    t0 = time.time()

    G, _, gan_history = train_gan(
        real_illicit,
        epochs=args.gan_epochs,
        device=device,
    )

    gan_time = time.time() - t0

    print(f"\nGAN trained in {gan_time:.1f}s")

    # -----------------------------------------
    # 6. Generate synthetic embeddings
    # -----------------------------------------

    n_synth = int(n_illicit * args.synth_mult)

    synth = generate_synthetic_embeddings(
        G,
        n_synth,
    )

    X = torch.cat(
        [real_licit, real_illicit, synth],
        dim=0,
    )

    y = torch.cat(
        [
            torch.zeros(n_licit, device=device),
            torch.ones(
                n_illicit + n_synth,
                device=device,
            ),
        ]
    )

    permutation = torch.randperm(
        len(y),
        device=device,
    )

    X = X[permutation]
    y = y[permutation]

    original_illicit_share = (
        100 * n_illicit / (n_illicit + n_licit)
    )

    augmented_illicit_share = (
        100 * y.mean().item()
    )

    print(
        f"Illicit share: "
        f"{original_illicit_share:.1f}% -> "
        f"{augmented_illicit_share:.1f}%"
    )

    # -----------------------------------------
    # 7. Train classifier head
    # -----------------------------------------

    head = nn.Linear(
        args.hidden,
        1,
    ).to(device)

    optimizer = torch.optim.Adam(
        head.parameters(),
        lr=0.01,
        weight_decay=1e-4,
    )

    best_val_f1 = -1.0
    best_state = None
    best_epoch = None

    head_history = []

    for epoch in range(1, args.head_epochs + 1):

        head.train()
        optimizer.zero_grad()

        logits = head(X).squeeze(-1)

        loss = F.binary_cross_entropy_with_logits(
            logits,
            y,
        )

        loss.backward()
        optimizer.step()

        # Evaluate on validation data
        if (
            epoch % 10 == 0
            or epoch == args.head_epochs
        ):

            val_metrics = evaluate_head(
                head,
                all_embeds,
                data.y,
                val_mask,
            )

            head_history.append(
                {
                    "epoch": epoch,
                    "loss": float(loss.item()),
                    **val_metrics,
                }
            )

            print(
                f"head epoch {epoch:3d} | "
                f"loss {loss.item():.4f} | "
                f"val F1 {val_metrics['f1_illicit']:.4f}"
            )

            if val_metrics["f1_illicit"] > best_val_f1:

                best_val_f1 = val_metrics["f1_illicit"]

                best_epoch = epoch

                best_state = copy.deepcopy(
                    head.state_dict()
                )

    # -----------------------------------------
    # 8. Evaluate best augmented classifier
    # -----------------------------------------

    head.load_state_dict(best_state)

    gan_metrics = evaluate_head(
        head,
        all_embeds,
        data.y,
        test_mask,
    )

    fpr_before = baseline["false_positive_rate"]
    fpr_after = gan_metrics["false_positive_rate"]

    if fpr_before > 0:
        fpr_reduction = (
            100 * (fpr_before - fpr_after) / fpr_before
        )
    else:
        fpr_reduction = None

    print("\nTest-set comparison:")

    print(
        f"{'metric':<22}"
        f"{'baseline':>12}"
        f"{'+ GAN':>12}"
    )

    for metric in [
        "f1_illicit",
        "precision_illicit",
        "recall_illicit",
        "false_positive_rate",
        "roc_auc",
    ]:
        print(
            f"{metric:<22}"
            f"{baseline[metric]:>12.4f}"
            f"{gan_metrics[metric]:>12.4f}"
        )

    if fpr_reduction is not None:
        print(
            f"\nFPR reduction: {fpr_reduction:.1f}%"
        )
    else:
        print(
            "\nFPR reduction: N/A "
            "(baseline FPR is zero)"
        )

    # -----------------------------------------
    # 9. Save classifier and metrics
    # -----------------------------------------

    for path in [args.head_out, args.metrics_out]:
        directory = os.path.dirname(path)

        if directory:
            os.makedirs(directory, exist_ok=True)

    torch.save(
        best_state,
        args.head_out,
    )

    results = {
        "dataset_meta": meta,
        "baseline_checkpoint": args.baseline_ckpt,
        "baseline_test_metrics": baseline,
        "gan_augmented_test_metrics": gan_metrics,
        "false_positive_rate_reduction_pct": fpr_reduction,
        "n_synthetic_generated": n_synth,
        "best_head_epoch": best_epoch,
        "best_validation_f1": best_val_f1,
        "gan_train_time_sec": gan_time,
        "gan_history": gan_history,
        "head_history": head_history,
    }

    with open(args.metrics_out, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved classifier: {args.head_out}")
    print(f"Saved metrics: {args.metrics_out}")


if __name__ == "__main__":
    main()