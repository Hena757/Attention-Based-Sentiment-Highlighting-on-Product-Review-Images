# Attention-Based Sentiment Highlighting on Product Review Images

**Team 10 — OCR + Attention**

Extract text from product review screenshots using OCR, then use an
attention-based text classifier to highlight which words most influenced the
sentiment prediction.

---

## Problem Statement

Product reviews are commonly shared as *screenshots* (e-commerce apps,
social media, support chats). Existing sentiment tools work on plain text, so
analysing a screenshot means manually transcribing it first. Even when a
sentiment label is produced, it is usually a black box — users cannot see
**which words** drove the decision, which is especially misleading for
negation-heavy reviews ("not bad", "never again"). This project closes both
gaps: it accepts a review screenshot, extracts the text with OCR, classifies
sentiment with an attention-based deep learning model, and **highlights the
words that received the highest attention** so the prediction is
interpretable.

## Abstract

The application implements an image-to-insight pipeline: uploaded review
screenshots are preprocessed with OpenCV, transcribed by Tesseract OCR, and
cleaned by a negation-preserving text preprocessor. A TensorFlow/Keras model
(Embedding → Bidirectional LSTM → additive attention → softmax) predicts the
sentiment and exposes per-token attention weights, which the interface renders
as word-level highlights and heatmaps. The repository currently contains the
complete application scaffold, preprocessing/OCR/visualisation code, the model
architecture, training and evaluation scripts, and unit tests. **Model
training and evaluation require a labelled dataset that is not bundled — no
pre-trained weights or metrics are claimed yet.**

## Objectives

1. Extract text from product review screenshots (OpenCV preprocessing +
   Tesseract OCR).
2. Clean and tokenise review text **without removing negation words**
   ("not", "never", "no").
3. Classify sentiment with an Embedding → BiLSTM → Attention → Dense network.
4. Surface per-token attention scores at inference time.
5. Highlight attention-weighted words in a Streamlit web interface.
6. Evaluate with accuracy, precision, recall, F1-score and a confusion
   matrix.
7. Keep modules separate, tested, and runnable from the repository root.

## Proposed Workflow

```text
Image Upload → Image Preprocessing → OCR Text Extraction → Text Preprocessing
→ Attention-Based Sentiment Classification → Word-Level Highlighting
→ Results Display
```

1. **Image Upload** — Streamlit file uploader (PNG/JPG/JPEG/BMP/WEBP).
2. **Image Preprocessing** — grayscale, upscale, denoise, adaptive
   threshold, morphology (`src/ocr.py`).
3. **OCR Text Extraction** — Tesseract via `pytesseract`, including
   word-level bounding boxes and confidence.
4. **Text Preprocessing** — HTML/URL/mention removal, contraction expansion
   that keeps negations, tokenisation, vocabulary encoding
   (`src/preprocessing.py`).
5. **Attention-Based Classification** — `src/attention_model.py` model,
   loaded through `src/predictor.py`.
6. **Word-Level Highlighting** — attention scores tint each word in HTML,
   plus a matplotlib heatmap and optional overlay on the original image
   (`src/visualization.py`).
7. **Results Display** — sentiment label, confidence, highlighted text,
   heatmap and class probabilities in the Streamlit app (`app/app.py`).

## Technology Stack

| Layer | Technology |
|---|---|
| Web interface | Streamlit |
| Deep learning | TensorFlow / Keras |
| OCR | Tesseract OCR engine + pytesseract wrapper |
| Image processing | OpenCV (`opencv-python-headless`), Pillow |
| Data & evaluation | NumPy, pandas, scikit-learn |
| Visualisation | matplotlib, seaborn |
| Testing | pytest |

## Repository Structure

```text
Attention-Based-Sentiment-Highlighting-on-Product-Review-Images/
│
├── app/
│   ├── __init__.py
│   └── app.py                 # Streamlit web application
│
├── src/
│   ├── __init__.py
│   ├── ocr.py                 # OpenCV preprocessing + Tesseract OCR
│   ├── preprocessing.py       # Negation-preserving text cleaning & vocab
│   ├── attention_model.py     # Embedding → BiLSTM → Attention → Dense
│   ├── predictor.py           # Model loading & inference with attention
│   └── visualization.py       # Word highlighting, heatmaps, overlays
│
├── training/
│   ├── train.py               # Train on a labelled review CSV
│   └── evaluate.py            # Accuracy/P/R/F1 + confusion matrix
│
├── notebooks/                 # Exploratory notebooks (see its README)
│
├── data/
│   ├── raw/                   # Put datasets here (git-ignored)
│   └── processed/             # Processed data (git-ignored)
│
├── models/                    # Trained artefacts land here (git-ignored)
│
├── tests/
│   ├── test_preprocessing.py
│   └── test_ocr.py
│
├── docs/
│   └── project_overview.md    # Problem, methodology, architecture
│
├── screenshots/               # App/evaluation screenshots for the report
│
├── .gitignore
├── requirements.txt
├── README.md
└── LICENSE
```

## Prerequisites

- **Python 3.10 – 3.12** (3.11 recommended for TensorFlow 2.16–2.19 wheels)
- **Tesseract OCR engine** — installed separately per OS; `pip install
  pytesseract` installs only the Python wrapper, **not** the engine:
  - Windows: [UB-Mannheim installer](https://github.com/UB-Mannheim/tesseract/wiki)
    (during setup tick *Additional language data* if needed; if `tesseract`
    is not on PATH set
    `pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"`)
  - macOS: `brew install tesseract`
  - Ubuntu/Debian: `sudo apt install tesseract-ocr`
- Git

## Setup Instructions

```bash
# 1. Clone the repository
git clone https://github.com/Hena757/Attention-Based-Sentiment-Highlighting-on-Product-Review-Images.git
cd Attention-Based-Sentiment-Highlighting-on-Product-Review-Images

# 2. Create and activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. (Optional) verify the OCR engine
tesseract --version
```

## Running the Application

```bash
# From the repository root:
streamlit run app/app.py
```

Then open the printed local URL (default http://localhost:8501), upload a
review screenshot, and the app will:

1. run OCR and show the extracted text,
2. show the preprocessed tokens,
3. **if a trained model exists** — show the sentiment, confidence,
   attention-highlighted review and attention heatmap.

> **Note:** step 3 requires a trained model. Without one, the app displays
> clear training instructions instead of predictions. This project does not
> bundle fake weights or fabricated outputs.

## Training

No dataset is bundled. Provide any labelled product-review CSV with a text
column and a sentiment label column (public options: Amazon product reviews,
Yelp reviews, or a Kaggle "product review" dataset), e.g.
`data/raw/reviews.csv` with columns `review,sentiment`:

```bash
python training/train.py \
    --data data/raw/reviews.csv \
    --text-column review \
    --label-column sentiment \
    --epochs 5
```

This saves `models/sentiment_attention.keras`, `models/vocab.json` and
`models/model_meta.json` (all git-ignored).

## Evaluation

```bash
python training/evaluate.py \
    --data data/raw/reviews_test.csv \
    --text-column review \
    --label-column sentiment
```

Prints accuracy, macro/per-class precision, recall and F1, the confusion
matrix, and saves the matrix figure to `screenshots/confusion_matrix.png`.

## Testing

```bash
# From the repository root:
pytest -v
```

- `tests/test_preprocessing.py` — always runs; verifies cleaning,
  **negation preservation**, contraction expansion, tokenisation, vocabulary
  building and encoding.
- `tests/test_ocr.py` — image loading and OpenCV preprocessing always run;
  the end-to-end OCR test is **skipped automatically** when the Tesseract
  engine is not installed.

## Current Implementation Status

| Feature | Status |
|---|---|
| Project structure & Git config | ✅ Implemented |
| Text preprocessing (negation-preserving) | ✅ Implemented & unit-tested |
| OpenCV image preprocessing | ✅ Implemented & unit-tested |
| Tesseract OCR integration (+ word boxes) | ✅ Implemented (end-to-end test auto-skips without engine) |
| Model architecture (BiLSTM + attention) | ✅ Implemented (definition only — **not yet trained**) |
| Attention-score visualisation (HTML, heatmap, overlay) | ✅ Implemented (code-level) |
| Streamlit app wiring | ✅ Implemented (prediction shown only after training) |
| Training script | ✅ Implemented (needs your dataset) |
| Evaluation script | ✅ Implemented (needs a trained model + dataset) |
| Trained model weights | ❌ **Not present** — must be trained locally |
| Measured accuracy / F1 metrics | ❌ **Not claimed** — none produced yet |
| Dataset download helper | ⏳ Planned |
| Screenshot overlay in the UI | ⏳ Planned |
| Transformer baseline comparison | ⏳ Planned |

## Limitations

- OCR quality depends on screenshot resolution and the Tesseract engine;
  stylised fonts, low contrast or heavy emoji use reduce accuracy.
- Attention weights are a *heuristic explanation*, not a causal guarantee.
- No pre-trained model is shipped: sentiment features only work after you
  train on a labelled dataset with `training/train.py`.
- The vocabulary and model must come from the same training run (the app
  loads both from `models/`).
- English-focused preprocessing; multilingual support is future work.

## Future Scope

- Overlay attention highlights directly on the uploaded screenshot in the UI.
- Dataset auto-download helpers and experiment tracking.
- Transformer-based baseline (e.g. DistilBERT) for comparison.
- Multilingual OCR and sentiment.
- Faithfulness studies for attention-based explanations.

## Contributors

**Team 10** — Attention-Based Sentiment Highlighting on Product Review
Images.

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE).
