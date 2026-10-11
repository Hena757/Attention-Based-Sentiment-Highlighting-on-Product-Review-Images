"""Train the attention-based sentiment model on a labelled review dataset.

Expected input: a CSV file with at least two columns - one free-text review
column and one sentiment label column (e.g. "positive"/"neutral"/"negative"
or 2/1/0). Example:

    python training/train.py \
        --data data/raw/reviews.csv \
        --text-column review \
        --label-column sentiment \
        --epochs 5

What it produces (all git-ignored, saved under models/):
    - sentiment_attention.keras   trained model
    - vocab.json                  word -> id vocabulary (shared with inference)
    - model_meta.json             max_length, label order, training config

No dataset or pre-trained weights are shipped with the repository; you must
provide your own labelled CSV (or download a public review dataset such as
the Amazon/Yelp/IMDB review datasets) before this script will run.
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

from src.attention_model import build_model
from src.preprocessing import build_vocab, preprocess_review, tokens_to_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Path to labelled CSV dataset")
    parser.add_argument("--text-column", default="review", help="CSV column containing review text")
    parser.add_argument("--label-column", default="sentiment", help="CSV column containing labels")
    parser.add_argument("--max-length", type=int, default=200, help="Padded sequence length")
    parser.add_argument("--max-vocab", type=int, default=20000, help="Maximum vocabulary size")
    parser.add_argument("--min-freq", type=int, default=2, help="Minimum token frequency for the vocab")
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--lstm-units", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--val-split", type=float, default=0.15, help="Fraction of training data used for validation")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "models"), help="Where to save artefacts")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def encode_dataset(token_lists, vocab, max_length: int) -> "np.ndarray":
    return np.asarray(
        [tokens_to_ids(tokens, vocab, max_length) for tokens in token_lists],
        dtype=np.int32,
    )


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    data_path = Path(args.data)
    if not data_path.is_file():
        sys.exit(
            f"Dataset not found: {data_path}\n"
            "Provide a labelled CSV (see README 'Datasets' section). "
            "No dataset is bundled with this repository."
        )

    df = pd.read_csv(data_path)
    for column in (args.text_column, args.label_column):
        if column not in df.columns:
            sys.exit(f"Column '{column}' not found in {data_path}. Available: {list(df.columns)}")

    df = df.dropna(subset=[args.text_column, args.label_column]).reset_index(drop=True)
    if df.empty:
        sys.exit("The dataset is empty after dropping missing values.")

    # Deterministic label encoding: sorted unique labels -> 0..K-1
    labels = sorted(df[args.label_column].unique())
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    y = df[args.label_column].map(label_to_id).to_numpy(dtype=np.int64)

    token_lists = [preprocess_review(text) for text in df[args.text_column].astype(str)]

    # Shuffle before the train/val split so it is stratification-friendly.
    order = rng.permutation(len(df))
    token_lists = [token_lists[i] for i in order]
    y = y[order]

    split = int(len(df) * (1.0 - args.val_split))
    train_tokens, val_tokens = token_lists[:split], token_lists[split:]
    y_train, y_val = y[:split], y[split:]

    if len(set(y_train)) < 2:
        sys.exit("Training split contains a single class; need at least two sentiment classes.")

    vocab = build_vocab(train_tokens, min_freq=args.min_freq, max_size=args.max_vocab)
    x_train = encode_dataset(train_tokens, vocab, args.max_length)
    x_val = encode_dataset(val_tokens, vocab, args.max_length) if val_tokens else None

    model = build_model(
        vocab_size=len(vocab),
        max_length=args.max_length,
        embedding_dim=args.embedding_dim,
        lstm_units=args.lstm_units,
        num_classes=len(labels),
        learning_rate=args.learning_rate,
    )
    model.summary()

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val) if x_val is not None and len(val_tokens) else None,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "sentiment_attention.keras"
    model.save(model_path)

    with open(output_dir / "vocab.json", "w", encoding="utf-8") as fh:
        json.dump(vocab, fh, ensure_ascii=False, indent=2)

    meta = {
        "max_length": args.max_length,
        "labels": labels,
        "label_to_id": label_to_id,
        "embedding_dim": args.embedding_dim,
        "lstm_units": args.lstm_units,
        "num_epochs": args.epochs,
        "dataset": str(data_path),
    }
    with open(output_dir / "model_meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)

    final_train_acc = history.history["accuracy"][-1]
    final_val_acc = (
        history.history["val_accuracy"][-1] if "val_accuracy" in history.history else None
    )
    print(f"\nSaved model     -> {model_path}")
    print(f"Saved vocab     -> {output_dir / 'vocab.json'}")
    print(f"Saved metadata  -> {output_dir / 'model_meta.json'}")
    print(f"Final training accuracy: {final_train_acc:.4f}")
    if final_val_acc is not None:
        print(f"Final validation accuracy: {final_val_acc:.4f}")


if __name__ == "__main__":
    main()
