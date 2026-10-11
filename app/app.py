"""Streamlit web application for attention-based sentiment highlighting.

Run from the repository root with:

    streamlit run app/app.py

Pipeline:  Image Upload -> Preprocessing -> OCR -> Text Preprocessing ->
           Attention-Based Sentiment Classification -> Word-Level
           Highlighting -> Results Display

The app is fully wired, but sentiment inference requires a trained model
produced by ``training/train.py``. When no model exists the app shows clear
setup instructions instead of fabricating predictions.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running via `streamlit run app/app.py` from the repository root:
# make the repo root importable so `src.*` resolves.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import streamlit as st

from src import ocr
from src.preprocessing import preprocess_review
from src.visualization import highlight_text_html, plot_attention_heatmap

st.set_page_config(
    page_title="Sentiment Highlighting | Team 10",
    page_icon="🧠",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def load_predictor():
    """Load the trained model once per app session (cached).

    Returns ``(predictor, error_message)``; exactly one is non-None.
    """
    try:
        from src.predictor import SentimentPredictor

        return SentimentPredictor(), None
    except FileNotFoundError as exc:
        return None, str(exc)
    except ImportError as exc:
        return None, f"TensorFlow is not installed: {exc}"


def main() -> None:
    st.title("Attention-Based Sentiment Highlighting")
    st.caption("Product review screenshots -> OCR -> attention-based sentiment -> word highlights")

    predictor, predictor_error = load_predictor()

    with st.sidebar:
        st.header("Status")
        st.write(f"Tesseract OCR engine: {'✅ available' if ocr.is_tesseract_available() else '❌ not installed'}")
        if predictor is not None:
            st.write("Sentiment model: ✅ loaded")
        else:
            st.write("Sentiment model: ❌ not trained")
        st.markdown(
            "---\n"
            "1. Install the Tesseract engine (see README)\n"
            "2. Train a model with `python training/train.py ...`\n"
            "3. Reload this page"
        )

    uploaded_file = st.file_uploader(
        "Upload a product review screenshot",
        type=["png", "jpg", "jpeg", "bmp", "webp"],
    )

    if uploaded_file is None:
        st.info("Upload a review screenshot to get started.")
        return

    image_bytes = uploaded_file.getvalue()
    try:
        import cv2

        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            st.error("Could not decode the uploaded image.")
            return
        st.image(image_bytes, caption="Uploaded screenshot", width=420)
    except Exception as exc:  # pragma: no cover - depends on upload contents
        st.error(f"Failed to read image: {exc}")
        return

    st.subheader("1. OCR text extraction")
    if not ocr.is_tesseract_available():
        st.error(ocr._TESSERACT_INSTALL_HELP)
        return

    with st.spinner("Running OCR..."):
        extracted_text = ocr.extract_text(image)
        words = ocr.extract_words(image)

    if not extracted_text.strip():
        st.warning("OCR returned no text. Try a sharper/larger screenshot.")
        return
    st.text_area("Extracted text", extracted_text, height=160)

    tokens = preprocess_review(extracted_text)
    st.write(f"**Preprocessed tokens ({len(tokens)}):**", " ".join(tokens))

    st.subheader("2. Sentiment prediction with attention highlighting")
    if predictor is None:
        st.warning("No trained model found. Run training first:")
        st.code(predictor_error or "", language="text")
        return

    with st.spinner("Classifying sentiment..."):
        result = predictor.predict(extracted_text)

    col1, col2, col3 = st.columns(3)
    col1.metric("Sentiment", result["label"].capitalize())
    col2.metric("Confidence", f"{result['confidence'] * 100:.1f}%")
    col3.metric("Tokens analysed", len(result["tokens"]))

    st.markdown("#### Highlighted review (attention-weighted)")
    st.markdown(
        highlight_text_html(result["tokens"], result["attention_scores"]),
        unsafe_allow_html=True,
    )

    st.markdown("#### Attention heatmap")
    fig = plot_attention_heatmap(result["tokens"], result["attention_scores"])
    st.pyplot(fig)

    with st.expander("Class probabilities"):
        st.json({k: round(v, 4) for k, v in result["probabilities"].items()})


if __name__ == "__main__":
    main()
