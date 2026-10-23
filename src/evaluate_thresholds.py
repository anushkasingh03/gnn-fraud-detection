"""
Compare GraphSAGE and GAN-augmented classifiers across decision thresholds.

Thresholds are selected using validation data only.
The held-out test set is used exclusively for final evaluation.

Run from the project root:
    python src/evaluate_thresholds.py --device cpu
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))

from data_loader import load_elliptic, time_based_split
from metrics import binary_metrics
from models import GraphSAGE


def get_metrics(probs, labels, threshold):
    return binary_metrics(probs, labels, threshold=threshold)


def threshold_candidates(probs):
    """
    Return all valid decision thresholds between 0 and 1.
    """
    unique = np.unique(np.asarray(probs, dtype=np.float64))

    return np.unique(
        np.concatenate([
            [0.0],
            unique,
            [1.0],
        ])
    )


def find_threshold_at_recall(probs, labels, target_recall):
    """
    Among thresholds meeting the target recall on validation data,
    choose the threshold with the lowest FPR.

    Break ties by selecting the highest threshold.
    """
    best_threshold = None
    best_fpr = float("inf")

    for threshold in threshold_candidates(probs):
        metrics = get_metrics(probs, labels, threshold)

        if metrics["recall_illicit"] + 1e-12 < target_recall:
            continue

        fpr = metrics["false_positive_rate"]

        if (
            fpr < best_fpr - 1e-12
            or (
                abs(fpr - best_fpr) <= 1e-12
                and (
                    best_threshold is None
                    or threshold > best_threshold
                )
            )
        ):
            best_threshold = float(threshold)
            best_fpr = fpr

    return best_threshold


def find_threshold_at_fpr(probs, labels, target_fpr):
    """
    Among thresholds meeting the target FPR on validation data,
    choose the threshold with the highest recall.

    Break ties by selecting the lowest threshold.
    """
    best_threshold = None
    best_recall = -1.0

    for threshold in threshold_candidates(probs):
        metrics = get_metrics(probs, labels, threshold)

        if metrics["false_positive_rate"] > target_fpr + 1e-12:
            continue

        recall = metrics["recall_illicit"]

        if (
            recall > best_recall + 1e-12
            or (
                abs(recall - best_recall) <= 1e-12
                and (
                    best_threshold is None
                    or threshold < best_threshold
                )
            )
        ):
            best_threshold = float(threshold)
            best_recall = recall

    return best_threshold


def fpr_reduction(baseline, candidate):
    before = baseline["false_positive_rate"]
    after = candidate["false_positive_rate"]

    if before == 0:
        return None

    return 100.0 * (before - after) / before


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
        "--gan_head_ckpt",
        default="models/real_gan_head.pt",
    )
    parser.add_argument(
        "--hidden",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "mps"],
        default="cpu",
    )
    parser.add_argument(
        "--out",
        default="outputs/real_threshold_analysis.json",
    )

    args = parser.parse_args()
    device = torch.device(args.device)

    print("Loading Elliptic dataset...")

    data, _ = load_elliptic(args.data_dir)
    train_mask, val_mask, test_mask = time_based_split(data)

    data = data.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)

    print("Loading trained GraphSAGE...")

    model = GraphSAGE(
        in_channels=data.x.shape[1],
        hidden_channels=args.hidden,
    ).to(device)

    model.load_state_dict(
        torch.load(
            args.baseline_ckpt,
            map_location=device,
            weights_only=True,
        )
    )
    model.eval()

    print("Loading GAN classifier head...")

    gan_head = nn.Linear(args.hidden, 1).to(device)
    gan_head.load_state_dict(
        torch.load(
            args.gan_head_ckpt,
            map_location=device,
            weights_only=True,
        )
    )
    gan_head.eval()

    print("Computing embeddings and probabilities...")

    with torch.no_grad():
        embeddings = model.embed(data.x, data.edge_index)

        baseline_probs = torch.sigmoid(
            model.classifier(embeddings).squeeze(-1)
        )

        gan_probs = torch.sigmoid(
            gan_head(embeddings).squeeze(-1)
        )

    def extract(probs, mask):
        return (
            probs[mask].cpu().numpy(),
            data.y[mask].cpu().numpy(),
        )

    baseline_val, y_val = extract(baseline_probs, val_mask)
    gan_val, _ = extract(gan_probs, val_mask)

    baseline_test, y_test = extract(baseline_probs, test_mask)
    gan_test, _ = extract(gan_probs, test_mask)

    # -------------------------------------------------
    # A. Baseline vs. GAN Augmentation
    # -------------------------------------------------

    print("\n=== A. Baseline vs. GAN Augmentation ===")

    baseline_fixed = get_metrics(baseline_test, y_test, 0.5)
    gan_fixed = get_metrics(gan_test, y_test, 0.5)

    print("Baseline:")
    print(json.dumps(baseline_fixed, indent=2))

    print("\nGAN:")
    print(json.dumps(gan_fixed, indent=2))

    print(
        "\nFPR reduction:",
        round(fpr_reduction(baseline_fixed, gan_fixed), 2),
        "%",
    )

    # -------------------------------------------------
    # B. Comparison at Matched Fraud Recall
    # -------------------------------------------------

    print("\n=== B. Comparison at Matched Fraud Recall ===")

    baseline_val_fixed = get_metrics(baseline_val, y_val, 0.5)

    target_recall = baseline_val_fixed["recall_illicit"]

    baseline_matched_threshold = find_threshold_at_recall(
        baseline_val,
        y_val,
        target_recall,
    )

    gan_matched_threshold = find_threshold_at_recall(
        gan_val,
        y_val,
        target_recall,
    )

    baseline_matched = get_metrics(
        baseline_test,
        y_test,
        baseline_matched_threshold,
    )

    gan_matched = get_metrics(
        gan_test,
        y_test,
        gan_matched_threshold,
    )

    print("Validation recall target:", round(target_recall, 4))

    print("Baseline threshold:", baseline_matched_threshold)
    print("GAN threshold:", gan_matched_threshold)

    print("\nBaseline test metrics:")
    print(json.dumps(baseline_matched, indent=2))

    print("\nGAN test metrics:")
    print(json.dumps(gan_matched, indent=2))

    print(
        "\nFPR reduction:",
        round(fpr_reduction(baseline_matched, gan_matched), 2),
        "%",
    )

    # -------------------------------------------------
    # C. Validation-Based Threshold Optimization
    # -------------------------------------------------

    print("\n=== C. Comparison at Matched False-Positive Rate ===")

    baseline_val_fpr = baseline_val_fixed["false_positive_rate"]

    target_fpr = baseline_val_fpr * 0.60

    gan_target_threshold = find_threshold_at_fpr(
        gan_val,
        y_val,
        target_fpr,
    )

    gan_target_val = get_metrics(
        gan_val,
        y_val,
        gan_target_threshold,
    )

    gan_target_test = get_metrics(
        gan_test,
        y_test,
        gan_target_threshold,
    )

    print("Baseline validation FPR:", baseline_val_fpr)
    print("Target validation FPR:", target_fpr)
    print("Selected GAN threshold:", gan_target_threshold)

    print("\nGAN validation metrics:")
    print(json.dumps(gan_target_val, indent=2))

    print("\nGAN test metrics:")
    print(json.dumps(gan_target_test, indent=2))

    print(
        "\nTest FPR reduction vs baseline at 0.5:",
        round(fpr_reduction(baseline_fixed, gan_target_test), 2),
        "%",
    )
    # -------------------------------------------------
    # D. Comparison at Matched False-Positive Rate
    # -------------------------------------------------

    print("\n=== D. Comparison at Matched False-Positive Rate ===")

    target_fpr = gan_target_val["false_positive_rate"]

    baseline_fpr_threshold = find_threshold_at_fpr(
        baseline_val,
        y_val,
        target_fpr,
    )

    gan_fpr_threshold = find_threshold_at_fpr(
        gan_val,
        y_val,
        target_fpr,
    )

    baseline_fpr_matched = get_metrics(
        baseline_test,
        y_test,
        baseline_fpr_threshold,
    )

    gan_fpr_matched = get_metrics(
        gan_test,
        y_test,
        gan_fpr_threshold,
    )

    print("Validation FPR target:", target_fpr)
    print("Baseline threshold:", baseline_fpr_threshold)
    print("GAN threshold:", gan_fpr_threshold)

    print("\nBaseline test metrics:")
    print(json.dumps(baseline_fpr_matched, indent=2))

    print("\nGAN test metrics:")
    print(json.dumps(gan_fpr_matched, indent=2))

    # -------------------------------------------------
    # Save results
    # -------------------------------------------------

    results = {
        "fixed_threshold": {
            "threshold": 0.5,
            "baseline": baseline_fixed,
            "gan": gan_fixed,
            "fpr_reduction_pct": fpr_reduction(
                baseline_fixed, gan_fixed
            ),
        },
        "matched_validation_recall": {
            "validation_recall_target": target_recall,
            "baseline_threshold": baseline_matched_threshold,
            "gan_threshold": gan_matched_threshold,
            "baseline_test": baseline_matched,
            "gan_test": gan_matched,
            "fpr_reduction_pct": fpr_reduction(
                baseline_matched, gan_matched
            ),
        },
        "target_40pct_fpr_reduction": {
            "baseline_validation_fpr": baseline_val_fpr,
            "target_validation_fpr": target_fpr,
            "gan_threshold": gan_target_threshold,
            "gan_validation": gan_target_val,
            "gan_test": gan_target_test,
            "test_fpr_reduction_pct": fpr_reduction(
                baseline_fixed, gan_target_test
            ),
        },
        "matched_validation_fpr": {
        "validation_fpr_target": target_fpr,
        "baseline_threshold": baseline_fpr_threshold,
        "gan_threshold": gan_fpr_threshold,
        "baseline_test": baseline_fpr_matched,
        "gan_test": gan_fpr_matched,
},
    }


    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    print("\nResults saved to:", args.out)


if __name__ == "__main__":
    main()