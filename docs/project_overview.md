# Project Overview

**Team 10 — Attention-Based Sentiment Highlighting on Product Review Images (OCR + Attention)**

## 1. Problem Statement

Product reviews are frequently shared as *screenshots* — in e-commerce apps,
social media, and customer-support chats. Standard sentiment-analysis tools
operate on plain text, so analysing a screenshot requires a human to first
transcribe it. Moreover, even when a sentiment label ("positive/negative") is
produced, it is a *black box*: users cannot see **which words** drove the
decision. This is especially problematic for reviews containing negation
("not bad", "never again"), where bag-of-words style systems frequently fail.

This project builds a pipeline that takes a **review screenshot**, extracts
its text with OCR, classifies sentiment with an **attention-based deep
learning model**, and **highlights the words that received the highest
attention**, making the prediction interpretable.

## 2. Objectives

1. Extract text from product review screenshots using Tesseract OCR with
   OpenCV-based image preprocessing.
2. Clean and tokenise review text while **preserving negation cues**
   ("not", "never", "no", ...).
3. Build a sentiment classifier using an Embedding → BiLSTM → Attention →
   Dense architecture in TensorFlow/Keras.
4. Expose per-token attention scores at inference time.
5. Visualise attention as word-level highlights in a Streamlit web app.
6. Evaluate the model with accuracy, precision, recall, F1-score and a
   confusion matrix.
7. Keep the codebase modular, testable, and reproducible.

## 3. Proposed Methodology

```
Image Upload
   │
   ▼
Image Preprocessing            OpenCV: grayscale, upscale, denoise,
   │                           adaptive threshold, morphology
   ▼
OCR Text Extraction            Tesseract via pytesseract (full text +
   │                           word boxes + confidence)
   ▼
Text Preprocessing             HTML/URL/mention removal, contraction
   │                           expansion (negations preserved), tokenising
   ▼
Attention-Based Classification Embedding → BiLSTM → additive attention
   │                           → softmax (negative / neutral / positive)
   ▼
Word-Level Highlighting        attention weights → per-token tint in HTML,
   │                           matplotlib heatmap, optional overlay on image
   ▼
Results Display                Streamlit UI: label, confidence, highlighted
                               text, heatmap, class probabilities
```

### 3.1 Image preprocessing
Screenshots are often compressed or low-resolution. The pipeline converts to
grayscale, upscales narrow images (Tesseract's accuracy improves sharply
above ~1800 px width), applies median-blur denoising, adaptive Gaussian
thresholding and a light morphological close.

### 3.2 Text preprocessing
Raw OCR text is cleaned of HTML, URLs, emails and mentions; contractions are
expanded ("isn't" → "is not") **so that negation words survive** rather than
being stripped by naive punctuation removal. A small vocabulary
(`word → id`, with `<PAD>`/`<UNK>`) is built at training time and stored as
`models/vocab.json` for identical inference-time encoding.

### 3.3 Model
An embedding layer (masking padding) feeds a Bidirectional LSTM that
maintains context from both directions. A Bahdanau-style **additive
attention** layer learns a weight for every token; the weighted representation
is pooled and passed to a softmax classifier. Because attention weights are
per-token, they double as an explanation signal.

### 3.4 Attention visualisation
Normalised attention scores tint each token's background in the web UI
(deeper colour = higher attention), a horizontal heatmap is rendered for
reports, and — using Tesseract word boxes — attention can be painted onto the
original screenshot.

## 4. Architecture & Module Responsibilities

| Module | Responsibility | Status |
|---|---|---|
| `app/app.py` | Streamlit UI wiring the full pipeline | Implemented |
| `src/ocr.py` | Image loading, OpenCV preprocessing, Tesseract text + word-box extraction | Implemented |
| `src/preprocessing.py` | Text cleaning, negation-preserving tokenisation, vocabulary & encoding | Implemented |
| `src/attention_model.py` | Embedding → BiLSTM → Attention → Dense model definition (TensorFlow/Keras) | Implemented |
| `src/predictor.py` | Load trained artefacts; return label, probabilities and attention scores | Implemented (requires trained model) |
| `src/visualization.py` | HTML word highlighting, attention heatmap, image overlay | Implemented |
| `training/train.py` | Train on a user-supplied labelled CSV; save model + vocab + metadata | Implemented (requires dataset) |
| `training/evaluate.py` | Accuracy, precision, recall, F1, confusion matrix on held-out data | Implemented (requires model + dataset) |
| `tests/` | Unit tests for preprocessing and OCR | Implemented |
| `docs/` | This document | Done |

## 5. Data

No dataset is shipped with the repository. Any labelled product-review CSV
with a text column and a sentiment label column works (e.g. Amazon product
reviews, Yelp reviews, or an IMDB-style dataset relabelled into
negative/neutral/positive). Datasets belong in `data/raw/` (git-ignored);
processed artefacts in `data/processed/`; trained weights in `models/`
(git-ignored).

## 6. Future Work

- End-to-end overlay of highlights on the *original* screenshot in the UI.
- Transformer-based baseline (e.g. a small BERT) for comparison.
- Dataset auto-download helper and experiment tracking.
- Multilingual OCR/sentiment support.
- Calibration of attention as a faithful explanation (attention is
  informative but not a guarantee of causality).
