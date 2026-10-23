# Financial Fraud Detection using GraphSAGE and GANs

A graph-based fraud detection system built with **PyTorch Geometric, GraphSAGE, and Generative Adversarial Networks (GANs)** to identify illicit Bitcoin transactions.

The project explores whether generating synthetic fraud embeddings can improve detection on imbalanced transaction data. It also includes threshold optimization, ONNX inference, and an AWS Lambda-compatible deployment pipeline.

## Results

Evaluated on the real **Elliptic Bitcoin Transaction Dataset** using a chronological train/validation/test split.

| Metric | GraphSAGE | GraphSAGE + GAN |
|---|---:|---:|
| Precision | 85.15% | 86.49% |
| Recall | 44.18% | 40.25% |
| F1-score | 58.18% | 54.94% |
| ROC-AUC | 88.02% | 87.94% |
| False-positive rate | 0.465% | 0.379% |

*Results at a classification threshold of 0.5.*

**Key findings:**
- GAN augmentation reduced the false-positive rate by **18.4%** at the default threshold, with a decrease in recall.
- Validation-based threshold optimization reduced the GraphSAGE baseline's false-positive rate by **46.9%**, from 0.465% to 0.246%.
- At comparable recall and false-positive operating points, the simpler GraphSAGE model outperformed the GAN-augmented classifier.
- Local ONNX inference achieved **0.075 ms median latency** and **100% prediction agreement** with PyTorch across 100 sampled transactions.

The 46.9% FPR reduction comes with a recall tradeoff (44.18% → 39.78%).

## Approach

### 1. GraphSAGE Baseline

Trained a three-layer GraphSAGE model to classify transactions using their features and graph relationships.

- 165 input features
- 128-dimensional hidden representations
- Class-weighted binary cross-entropy
- Validation-based model selection

### 2. GAN Augmentation

To address class imbalance, I trained a GAN on illicit transaction embeddings extracted from GraphSAGE.

The generated embeddings were combined with real training embeddings to train a new classification head while keeping the GraphSAGE backbone frozen.

I also implemented random oversampling as a comparison.

### 3. Threshold Optimization

Instead of relying only on the default classification threshold of 0.5, I selected thresholds using validation data and evaluated them on a held-out temporal test set.

This helped measure the tradeoff between catching fraudulent transactions and incorrectly flagging legitimate ones.

### 4. ONNX Inference

Exported the trained GraphSAGE model to ONNX and built a lightweight inference handler for transaction subgraphs.

Local benchmarks showed sub-millisecond inference, with predictions matching the PyTorch implementation.

An AWS Lambda-compatible Docker container and handler are included. **Cloud deployment is not yet complete.**

## Dataset

The [Elliptic Bitcoin Transaction Dataset](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) contains:

- **203,769 transactions** represented as graph nodes
- **234,355 transaction relationships**
- **165 features** per transaction
- **4,545 labeled illicit transactions**
- **42,019 labeled licit transactions**

The dataset was split chronologically:

| Split | Time Steps |
|---|---|
| Train | 1–34 |
| Validation | 35–39 |
| Test | 40–49 |

Unlabeled transactions were retained as graph context but excluded from supervised evaluation.

## Project Structure

```text
gnn-fraud-detection/
├── src/
│   ├── data_loader.py
│   ├── models.py
│   ├── gan.py
│   ├── train_baseline.py
│   ├── train_gan_augmented.py
│   ├── train_oversampled.py
│   └── evaluate_thresholds.py
├── lambda/
│   ├── Dockerfile
│   ├── export_to_onnx.py
│   ├── lambda_handler.py
│   └── benchmark_inference.py
├── notebooks/
├── outputs/
├── models/
├── requirements.txt
└── README.md
```

## Getting Started

**1. Clone the repository and install dependencies**

```bash
git clone <repository-url>
cd gnn-fraud-detection

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**2. Download the Elliptic dataset**

Place the dataset CSV files in `data/elliptic_real/`.

**3. Train the baseline**

```bash
python src/train_baseline.py --data_dir data/elliptic_real --epochs 150 --device cpu
```

**4. Train the GAN-augmented classifier**

```bash
python src/train_gan_augmented.py --gan_epochs 500 --head_epochs 150 --device cpu
```

**5. Evaluate threshold optimization**

```bash
python src/evaluate_thresholds.py --device cpu
```

Detailed experiment results are available in the `outputs/` directory.

## Tech Stack

**Python · PyTorch · PyTorch Geometric · GraphSAGE · GANs · scikit-learn · ONNX Runtime · Docker · AWS Lambda**
