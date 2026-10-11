"""Load a trained attention model and run sentiment inference.

Responsibilities
----------------
- Locate and load the trained model + vocabulary produced by
  ``training/train.py`` (``models/sentiment_attention.keras`` and
  ``models/vocab.json``).
- Preprocess raw review text exactly the way training did.
- Return the sentiment label, class probabilities AND the per-token attention
  scores, so the visualization module can highlight the influential words.

IMPORTANT: this module refuses to run when no trained artefacts exist. It
raises ``FileNotFoundError`` with setup instructions - it never invents
predictions or random weights.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from src.preprocessing import PAD_ID, preprocess_review, tokens_to_ids

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = REPO_ROOT / "models" / "sentiment_attention.keras"
DEFAULT_VOCAB_PATH = REPO_ROOT / "models" / "vocab.json"
DEFAULT_META_PATH = REPO_ROOT / "models" / "model_meta.json"

_MISSING_ARTEFACTS_MSG = (
    "No trained model found.\n"
    "Expected files:\n"
    "  - {model}\n"
    "  - {vocab}\n"
    "Train a model first, e.g.:\n"
    "  python training/train.py --data data/raw/reviews.csv --text-column review "
    "--label-column sentiment\n"
    "See the README 'Training' section for details."
)


def _lazy_tf():
    """Import TensorFlow only when needed (keeps OCR/preprocessing light)."""
    try:
        import tensorflow as tf  # noqa: WPS433
        from src.attention_model import custom_objects  # noqa: WPS433
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "TensorFlow is required for inference. Install dependencies with:\n"
            "  pip install -r requirements.txt"
        ) from exc
    return tf, custom_objects


class SentimentPredictor:
    """Predict sentiment for review text and expose attention scores."""

    def __init__(
        self,
        model_path: Union[str, Path] = DEFAULT_MODEL_PATH,
        vocab_path: Union[str, Path] = DEFAULT_VOCAB_PATH,
        meta_path: Union[str, Path] = DEFAULT_META_PATH,
    ):
        self.model_path = Path(model_path)
        self.vocab_path = Path(vocab_path)
        self.meta_path = Path(meta_path)

        if not self.model_path.is_file() or not self.vocab_path.is_file():
            raise FileNotFoundError(
                _MISSING_ARTEFACTS_MSG.format(model=self.model_path, vocab=self.vocab_path)
            )

        tf, custom_objects = _lazy_tf()
        self._tf = tf

        self.model = tf.keras.models.load_model(
            str(self.model_path), custom_objects=custom_objects()
        )
        with open(self.vocab_path, "r", encoding="utf-8") as fh:
            self.vocab: Dict[str, int] = json.load(fh)

        self.meta: Dict = {}
        if self.meta_path.is_file():
            with open(self.meta_path, "r", encoding="utf-8") as fh:
                self.meta = json.load(fh)

        self.max_length: int = int(
            self.meta.get("max_length", self.model.input_shape[1])
        )
        self.labels: List[str] = list(
            self.meta.get("labels", ["negative", "neutral", "positive"])
        )

        from src.attention_model import build_attention_probe  # noqa: WPS433

        self._probe = build_attention_probe(self.model)

    def _encode(self, text: str) -> Tuple["np.ndarray", List[str]]:
        tokens = preprocess_review(text)
        ids = tokens_to_ids(tokens, self.vocab, self.max_length)
        return np.asarray([ids], dtype=np.int32), tokens

    def predict(self, text: str) -> Dict:
        """Predict sentiment for a single review string.

        Returns a dict with tokens, class probabilities, the predicted label
        and per-token attention scores (aligned with ``tokens``, padding
        stripped).
        """
        encoded, tokens = self._encode(text)
        probs_arr, attn_arr = self._probe.predict(encoded, verbose=0)

        probabilities = probs_arr[0]
        attention = attn_arr[0]

        valid_length = min(len(tokens), self.max_length)
        attention_scores = [float(a) for a in attention[:valid_length]]

        class_id = int(np.argmax(probabilities))
        return {
            "text": text,
            "tokens": tokens[:valid_length],
            "label": self.labels[class_id] if class_id < len(self.labels) else str(class_id),
            "class_id": class_id,
            "confidence": float(probabilities[class_id]),
            "probabilities": {
                self.labels[i] if i < len(self.labels) else str(i): float(p)
                for i, p in enumerate(probabilities)
            },
            "attention_scores": attention_scores,
        }

    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """Run :meth:`predict` over a list of review strings."""
        return [self.predict(t) for t in texts]


_default_predictor: Optional[SentimentPredictor] = None


def get_predictor() -> SentimentPredictor:
    """Return a process-wide cached predictor (loads artefacts once)."""
    global _default_predictor
    if _default_predictor is None:
        _default_predictor = SentimentPredictor()
    return _default_predictor
