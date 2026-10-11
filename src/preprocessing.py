"""Text preprocessing for product review sentiment analysis.

Responsibilities
----------------
- Clean raw review text (HTML, URLs, mentions, punctuation, extra whitespace)
  while deliberately PRESERVING negation words ("not", "never", "no", ...)
  because they invert sentiment and are critical for an attention-based model.
- Tokenize cleaned text into a list of word tokens.
- Build a small, version-independent vocabulary (word -> integer id) that is
  shared between training (training/train.py) and inference (src/predictor.py).

No deep-learning framework is imported here on purpose: this module must stay
lightweight so it can be unit-tested without TensorFlow installed.
"""

from __future__ import annotations

import html
import re
import unicodedata
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence

# Special token ids used everywhere in the project.
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_ID = 0
UNK_ID = 1

#: Words that flip or negate sentiment. Never strip or drop these.
NEGATION_WORDS = frozenset(
    {
        "not",
        "no",
        "never",
        "neither",
        "nobody",
        "nothing",
        "nowhere",
        "nor",
        "cannot",
        "without",
        "hardly",
        "barely",
        "scarcely",
        "don't",
        "doesn't",
        "didn't",
        "isn't",
        "aren't",
        "wasn't",
        "weren't",
        "won't",
        "wouldn't",
        "shouldn't",
        "couldn't",
        "haven't",
        "hasn't",
        "hadn't",
        "mustn't",
        "ain't",
    }
)

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"(?:https?://|www\.)\S+")
_EMAIL_RE = re.compile(r"\b\S+@\S+\.\S+\b")
_MENTION_RE = re.compile(r"@\w+")
# Keep apostrophes inside contractions (don't, isn't) but drop other punctuation.
_NON_WORD_RE = re.compile(r"[^0-9a-zA-Z'\s]+")
_WHITESPACE_RE = re.compile(r"\s+")
_WORD_RE = re.compile(r"[0-9a-zA-Z']+")


def expand_contractions(text: str) -> str:
    """Expand common English contractions so "isn't" -> "is not".

    Expanding keeps the negation word itself ("not", "never", "no") intact,
    which is exactly what the sentiment model needs to learn from.
    """
    replacements = {
        "can't": "cannot",
        "won't": "will not",
        "shan't": "shall not",
        "ain't": "is not",
    }
    # Irregular forms first, so the generic n't rule does not mangle them.
    for contraction, expansion in replacements.items():
        text = re.sub(re.escape(contraction), expansion, text, flags=re.IGNORECASE)
    # Generic rules: 'll -> will, 're -> are, 've -> have, 'd -> would, 'm -> am,
    # 's -> is, and n't -> not ("don't" -> "do not", "isn't" -> "is not").
    text = re.sub(r"\b(\w+?)'ll\b", r"\1 will", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(\w+?)'re\b", r"\1 are", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(\w+?)'ve\b", r"\1 have", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(\w+?)'d\b", r"\1 would", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(\w+?)'m\b", r"\1 am", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(\w+?)'s\b", r"\1 is", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(\w+?)n't\b", r"\1 not", text, flags=re.IGNORECASE)
    return text


def clean_text(text: str, *, lowercase: bool = True, expand: bool = True) -> str:
    """Clean a raw review string and return normalised text.

    Parameters
    ----------
    text:
        Raw review text, possibly containing HTML, URLs or emoji noise.
    lowercase:
        Lower-case the text (recommended for the sentiment model).
    expand:
        Expand contractions such as "isn't" -> "is not", which preserves the
        negation signal instead of deleting it.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)

    text = html.unescape(text)                      # &amp; -> &
    text = unicodedata.normalize("NFKC", text)      # normalise unicode
    text = _HTML_TAG_RE.sub(" ", text)              # strip HTML tags
    text = _URL_RE.sub(" ", text)                   # strip URLs
    text = _EMAIL_RE.sub(" ", text)                 # strip emails
    text = _MENTION_RE.sub(" ", text)               # strip @mentions
    if expand:
        text = expand_contractions(text)
    if lowercase:
        text = text.lower()
    text = _NON_WORD_RE.sub(" ", text)              # drop remaining punctuation/digits symbols
    text = _WHITESPACE_RE.sub(" ", text).strip()    # collapse whitespace
    return text


def tokenize(text: str, *, clean: bool = True, lowercase: bool = True) -> List[str]:
    """Return a list of word tokens for ``text``.

    With ``clean=True`` the text is cleaned first; negation words survive
    cleaning and therefore survive tokenisation.
    """
    if clean:
        text = clean_text(text, lowercase=lowercase)
    return _WORD_RE.findall(text.lower() if lowercase and not clean else text)


def preprocess_review(text: str, *, lowercase: bool = True) -> List[str]:
    """Full preprocessing pipeline: raw review -> model-ready tokens."""
    return tokenize(text, clean=True, lowercase=lowercase)


def contains_negation(tokens: Sequence[str]) -> bool:
    """Return True if any token is a known negation word."""
    return any(tok in NEGATION_WORDS for tok in tokens)


def build_vocab(
    token_lists: Iterable[Sequence[str]],
    *,
    min_freq: int = 1,
    max_size: Optional[int] = None,
) -> Dict[str, int]:
    """Build a word->id vocabulary from an iterable of token lists.

    Ids 0 and 1 are reserved for ``<PAD>`` and ``<UNK>``. Remaining words are
    ordered by frequency (descending), then alphabetically for determinism.
    The returned dict is JSON-serialisable and saved next to the trained model.
    """
    counter: Counter = Counter()
    for tokens in token_lists:
        counter.update(tokens)

    vocab: Dict[str, int] = {PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID}
    ordered = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
    for word, freq in ordered:
        if freq < min_freq:
            continue
        if max_size is not None and len(vocab) >= max_size:
            break
        vocab[word] = len(vocab)
    return vocab


def tokens_to_ids(
    tokens: Sequence[str],
    vocab: Dict[str, int],
    max_length: int,
) -> List[int]:
    """Map tokens to integer ids and pad/truncate to exactly ``max_length``."""
    ids = [vocab.get(tok, UNK_ID) for tok in tokens[:max_length]]
    if len(ids) < max_length:
        ids.extend([PAD_ID] * (max_length - len(ids)))
    return ids
