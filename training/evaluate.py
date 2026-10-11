"""Evaluate a trained attention model on a held-out labelled dataset.

Reports accuracy, precision, recall, F1 (macro and per-class) and a
confusion-matrix figure saved under screenshots/.

Example:

    python training/evaluate.py \
        --data data/raw/reviews_test.csv \
        --text-column review \
        --label-column sentiment

Requires artefacts produced by ``training/train.py`` under models/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.predictor import (
    DEFAULT_META_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_VOCAB_PATH,
    SentimentPredictor,
)
from src.preprocessing import preprocess_review, tokens_to_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Path to labelled CSV dataset")
    parser.add_argument("--text-column", default="review")
    parser.add_argument("--label-column", default="sentiment")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--vocab", default=str(DEFAULT_VOCAB_PATH))
    parser.add_argument("--meta", default=str(DEFAULT_META_PATH))
    parser.add_argument(
        "--confusion-matrix-out",
        default=str(REPO_ROOT / "screenshots" / "confusion_matrix.png"),
        help="Where to save the confusion-matrix plot",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        predictor = SentimentPredictor(
            model_path=args.model, vocab_path=args.vocab, meta_path=args.meta
        )
    except FileNotFoundError as exc:
        sys.exit(str(exc))

    data_path = Path(args.data)
    if not data_path.is_file():
        sys.exit(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path).dropna(subset=[args.text_column, args.label_column])
    if df.empty:
        sys.exit("The evaluation dataset is empty.")

    labels_order = predictor.labels  # e.g. ["negative", "neutral", "positive"]
    label_to_id = {label: idx for idx, label in enumerate(labels_order)}

    unknown = sorted(set(df[args.label_column]) - set(label_to_id))
    if unknown:
        sys.exit(
            f"Dataset contains labels the model was not trained on: {unknown}. "
            f"Model labels: {labels_order}"
        )

    y_true = df[args.label_column].map(label_to_id).to_numpy()
    y_pred = []
    for text in df[args.text_column].astype(str):
        ids, _ = predictor._encode(text)  # reuse predictor encoding
        probs = predictor.model.predict(ids, verbose=0)[0]
        y_pred.append(int(np.argmax(probs)))
    y_pred = np.asarray(y_pred)

    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    accuracy = accuracy_score(y_true, y_pred)
    macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print(f"Evaluating on {len(df)} samples from {data_path}\n")
    print(f"Accuracy:            {accuracy:.4f}")
    print(f"Macro precision:     {macro_precision:.4f}")
    print(f"Macro recall:        {macro_recall:.4f}")
    print(f"Macro F1:            {macro_f1:.4f}\n")
    print(classification_report(y_true, y_pred, target_names=labels_order, zero_division=0))
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)

    # Save the confusion matrix figure for the report.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels_order,
        yticklabels=labels_order,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()

    out_path = Path(args.confusion_matrix_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nConfusion matrix figure saved -> {out_path}")


if __name__ == "__main__":
    main()
