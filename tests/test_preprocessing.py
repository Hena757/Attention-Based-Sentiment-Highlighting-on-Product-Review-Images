"""Unit tests for src.preprocessing.

Run from the repository root:

    pytest tests/test_preprocessing.py -v
"""

import pytest

from src.preprocessing import (
    NEGATION_WORDS,
    PAD_ID,
    UNK_ID,
    build_vocab,
    clean_text,
    contains_negation,
    expand_contractions,
    preprocess_review,
    tokenize,
    tokens_to_ids,
)


class TestCleanText:
    def test_lowercases_and_strips(self):
        assert clean_text("  GREAT Product!! ") == "great product"

    def test_preserves_negation_words(self):
        cleaned = clean_text("This is not good. I would never buy it again, no way.")
        tokens = cleaned.split()
        for word in ("not", "never", "no"):
            assert word in tokens

    def test_removes_urls(self):
        cleaned = clean_text("great item check https://example.com/xyz now")
        assert "https" not in cleaned
        assert "example.com" not in cleaned

    def test_removes_html_tags(self):
        cleaned = clean_text("<p>Excellent <b>quality</b></p>")
        assert cleaned == "excellent quality"

    def test_removes_mentions(self):
        cleaned = clean_text("thanks @amazon support helped me")
        assert "@amazon" not in cleaned

    def test_handles_none(self):
        assert clean_text(None) == ""

    def test_collapses_whitespace(self):
        assert clean_text("too    many\n\nspaces") == "too many spaces"


class TestContractions:
    def test_isnt_becomes_is_not(self):
        assert expand_contractions("it isn't cheap") == "it is not cheap"

    def test_dont_becomes_do_not(self):
        assert expand_contractions("I don't like it") == "I do not like it"

    def test_dont_is_case_insensitive(self):
        assert expand_contractions("Don't waste money") == "Do not waste money"

    def test_wont_becomes_will_not(self):
        assert "will not" in expand_contractions("I won't return it")

    def test_negation_survives_expansion(self):
        tokens = preprocess_review("This isn't a good product")
        assert "not" in tokens
        assert "isn't" not in tokens


class TestTokenize:
    def test_tokenize_returns_words(self):
        assert tokenize("battery life is amazing") == ["battery", "life", "is", "amazing"]

    def test_tokenize_keeps_negation(self):
        tokens = tokenize("not bad at all")
        assert "not" in tokens

    def test_preprocess_review_pipeline(self):
        tokens = preprocess_review("<b>Very</b> good! https://spam.example")
        assert tokens == ["very", "good"]


class TestNegation:
    def test_contains_negation_true(self):
        assert contains_negation(["this", "is", "not", "good"])

    def test_contains_negation_false(self):
        assert not contains_negation(["this", "is", "good"])

    def test_negation_words_are_lowercase(self):
        assert all(word == word.lower() for word in NEGATION_WORDS)


class TestVocabAndEncoding:
    def test_build_vocab_special_ids(self):
        vocab = build_vocab([["good", "bad"], ["good", "good"]])
        assert vocab["<PAD>"] == PAD_ID
        assert vocab["<UNK>"] == UNK_ID
        assert vocab["good"] > PAD_ID

    def test_build_vocab_frequency_order(self):
        vocab = build_vocab([["a", "a", "b"]])
        # "a" (freq 2) should come before "b" (freq 1)
        assert vocab["a"] < vocab["b"]

    def test_build_vocab_min_freq(self):
        vocab = build_vocab([["a", "a", "b"]], min_freq=2)
        assert "a" in vocab
        assert "b" not in vocab

    def test_tokens_to_ids_pads(self):
        vocab = build_vocab([["good", "bad", "ugly"]])
        ids = tokens_to_ids(["good"], vocab, max_length=4)
        assert len(ids) == 4
        assert ids[0] == vocab["good"]
        assert ids[1:] == [PAD_ID, PAD_ID, PAD_ID]

    def test_tokens_to_ids_truncates(self):
        vocab = build_vocab([["a", "b", "c", "d"]])
        ids = tokens_to_ids(["a", "b", "c", "d"], vocab, max_length=2)
        assert len(ids) == 2

    def test_unknown_token_maps_to_unk(self):
        vocab = build_vocab([["known"]])
        ids = tokens_to_ids(["mystery"], vocab, max_length=1)
        assert ids == [UNK_ID]

    def test_empty_input(self):
        assert preprocess_review("") == []
        assert clean_text("") == ""
