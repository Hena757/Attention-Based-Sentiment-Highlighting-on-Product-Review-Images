"""Visualize attention scores by highlighting influential review words.

Responsibilities
----------------
- Turn (token, attention-score) pairs into HTML where each word's background
  intensity reflects its attention weight (this is what the Streamlit app
  renders).
- Provide a matplotlib heatmap for reports/notebooks.
- Optionally paint attention-weighted boxes back onto the original OCR
  screenshot using the word boxes produced by ``src.ocr.extract_words``.

This module contains no model logic - it only visualises scores it is given.
"""

from __future__ import annotations

import html
from typing import List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")  # headless-safe default; notebooks can still switch backend
import matplotlib.pyplot as plt
import numpy as np

from src.ocr import OcrWord


def _normalize(scores: Sequence[float]) -> List[float]:
    """Min-max normalise scores to [0, 1]; constant input maps to zeros."""
    arr = np.asarray(scores, dtype=float)
    if arr.size == 0:
        return []
    lo, hi = float(arr.min()), float(arr.max())
    if np.isclose(hi, lo):
        return [0.0] * len(arr)
    return [float((s - lo) / (hi - lo)) for s in arr]


def highlight_text_html(
    tokens: Sequence[str],
    attention_scores: Sequence[float],
    *,
    threshold: Optional[float] = None,
    base_color: str = "255, 235, 59",
    show_legend: bool = True,
) -> str:
    """Return an HTML fragment highlighting tokens by attention weight.

    Parameters
    ----------
    tokens, attention_scores:
        Must be the same length (the predictor guarantees this).
    threshold:
        When given, only tokens whose *normalised* attention >= threshold are
        highlighted; others render plain. Without it every token is tinted
        proportionally.
    base_color:
        RGB triple used for the highlight tint (yellow by default).

    The returned string is safe to render with Streamlit's
    ``st.markdown(..., unsafe_allow_html=True)`` or any HTML sink.
    """
    if len(tokens) != len(attention_scores):
        raise ValueError(
            f"tokens ({len(tokens)}) and attention_scores "
            f"({len(attention_scores)}) must have the same length"
        )

    norm = _normalize(attention_scores)
    parts: List[str] = []
    for token, score in zip(tokens, norm):
        escaped = html.escape(token)
        if threshold is not None and score < threshold:
            parts.append(escaped)
        else:
            alpha = 0.15 + 0.85 * score  # keep faint words still visible
            parts.append(
                f'<span style="background-color: rgba({base_color}, {alpha:.3f}); '
                f'padding: 1px 3px; margin: 0 1px; border-radius: 3px;">{escaped}</span>'
            )
    body = " ".join(parts)

    legend = ""
    if show_legend:
        legend = (
            '<div style="margin-top: 8px; font-size: 0.85em; color: gray;">'
            "Attention: <span style=\"background-color: rgba(255, 235, 59, 0.15);\">low</span>"
            " \u2192 <span style=\"background-color: rgba(255, 235, 59, 1.0);\">high</span>"
            "</div>"
        )
    return f"<div style='line-height: 2.1;'>{body}</div>{legend}"


def plot_attention_heatmap(
    tokens: Sequence[str],
    attention_scores: Sequence[float],
    *,
    title: str = "Token attention",
    figsize: tuple = (10, 0.45),
):
    """Create a horizontal attention heatmap (one row per token).

    Returns the matplotlib ``Figure``; the caller decides whether to ``show``
    or ``savefig`` it.
    """
    if len(tokens) != len(attention_scores):
        raise ValueError("tokens and attention_scores must have the same length")
    if not tokens:
        raise ValueError("Cannot plot an empty attention sequence")

    norm = np.asarray(_normalize(attention_scores)).reshape(-1, 1)
    height = max(figsize[1] * len(tokens), 2.0)
    fig, ax = plt.subplots(figsize=(figsize[0], height))
    ax.imshow(norm, aspect="auto", cmap="YlOrRd", vmin=0.0, vmax=1.0)
    ax.set_yticks(range(len(tokens)))
    ax.set_yticklabels(list(tokens), fontsize=9)
    ax.set_xticks([])
    ax.set_title(title, fontsize=11)
    ax.grid(False)
    fig.tight_layout()
    return fig


def draw_attention_on_image(
    image: np.ndarray,
    words: Sequence[OcrWord],
    attention_scores: Sequence[float],
    *,
    title: Optional[str] = None,
):
    """Paint attention-weighted rectangles onto the original screenshot.

    ``words`` come from :func:`src.ocr.extract_words`; ``attention_scores``
    must be aligned with ``words`` (same order/length). Returns a matplotlib
    ``Figure`` with the annotated image.
    """
    if len(words) != len(attention_scores):
        raise ValueError("words and attention_scores must have the same length")

    norm = _normalize(attention_scores)
    canvas = image.copy()
    if canvas.ndim == 2:
        canvas = cv2_gray_to_rgb(canvas)

    for word, score in zip(words, norm):
        color = _score_to_rgb(score)
        cv2 = _cv2()
        cv2.rectangle(
            canvas,
            (word.left, word.top),
            (word.left + word.width, word.top + word.height),
            color,
            2,
        )

    fig, ax = plt.subplots(figsize=(10, max(canvas.shape[0] / 100.0, 3.0)))
    ax.imshow(canvas)
    ax.axis("off")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


def _cv2():
    import cv2  # local import keeps matplotlib-only usage light

    return cv2


def cv2_gray_to_rgb(image: np.ndarray) -> np.ndarray:
    return _cv2().cvtColor(image, _cv2().COLOR_GRAY2RGB)


def _score_to_rgb(score: float):
    """Map a 0..1 attention score to a blue(high) -> red(low) RGB color."""
    r = int(255 * (1.0 - score))
    b = int(255 * score)
    return (r, 64, b)
