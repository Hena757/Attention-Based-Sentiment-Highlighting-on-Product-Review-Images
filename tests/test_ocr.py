"""Unit tests for src.ocr.

Run from the repository root:

    pytest tests/test_ocr.py -v

The end-to-end OCR test requires the Tesseract engine to be installed
separately (see src/ocr.py docstring). It is skipped automatically when the
engine is not available, so the rest of the suite still passes.
"""

import shutil

import pytest

cv2 = pytest.importorskip("cv2", reason="opencv-python-headless is not installed")
np = pytest.importorskip("numpy", reason="numpy is not installed")

from src.ocr import (  # noqa: E402
    OcrWord,
    extract_text,
    extract_words,
    is_tesseract_available,
    load_image,
    preprocess_image,
)

TESSERACT_INSTALLED = shutil.which("tesseract") is not None


def _make_test_image(text: str = "NOT BAD AT ALL", size=(600, 120)):
    """Create a synthetic screenshot-like image with big readable text."""
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", size, color="white")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 44)
    except OSError:
        font = ImageFont.load_default()
    draw.text((20, 30), text, fill="black", font=font)
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


class TestLoadImage:
    def test_load_from_numpy_array(self):
        arr = np.zeros((10, 10, 3), dtype=np.uint8)
        loaded = load_image(arr)
        assert loaded.shape == (10, 10, 3)

    def test_load_from_grayscale_array(self):
        arr = np.zeros((10, 10), dtype=np.uint8)
        loaded = load_image(arr)
        assert loaded.ndim == 2

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_image(tmp_path / "does_not_exist.png")

    def test_empty_image_raises(self):
        with pytest.raises(ValueError):
            load_image(np.array([], dtype=np.uint8))


class TestPreprocessImage:
    def test_returns_grayscale(self):
        image = _make_test_image()
        processed = preprocess_image(image)
        assert processed.ndim == 2

    def test_upscales_small_images(self):
        image = _make_test_image(size=(200, 60))
        processed = preprocess_image(image, upscale=True, max_width=1000)
        assert processed.shape[1] >= 1000

    def test_does_not_upscale_beyond_max_width(self):
        image = _make_test_image(size=(2200, 120))
        processed = preprocess_image(image, upscale=True, max_width=1800)
        # Original is already wider than max_width: no scaling applied.
        assert processed.shape[1] == 2200

    def test_upscale_disabled(self):
        image = _make_test_image(size=(200, 60))
        processed = preprocess_image(image, upscale=False)
        assert processed.shape[1] == 200

    def test_accepts_grayscale_input(self):
        image = _make_test_image()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        processed = preprocess_image(gray)
        assert processed.ndim == 2

    def test_output_is_binary_like(self):
        image = _make_test_image()
        processed = preprocess_image(image)
        unique = np.unique(processed)
        assert set(unique).issubset({0, 255})


class TestOcrWordDataclass:
    def test_box_property(self):
        word = OcrWord(text="great", left=10, top=20, width=50, height=30, confidence=95.0)
        assert word.box == (10, 20, 50, 30)


@pytest.mark.skipif(
    not TESSERACT_INSTALLED,
    reason="Tesseract OCR engine is not installed on this system",
)
class TestEndToEndOcr:
    def test_extract_text_finds_words(self):
        image = _make_test_image(text="NOT BAD AT ALL")
        text = extract_text(image)
        assert text.strip()
        assert "bad" in text.lower()

    def test_extract_words_returns_boxes(self):
        image = _make_test_image(text="GREAT PRODUCT")
        words = extract_words(image)
        assert len(words) > 0
        first = words[0]
        assert isinstance(first, OcrWord)
        assert first.width > 0 and first.height > 0


def test_is_tesseract_available_returns_bool():
    assert isinstance(is_tesseract_available(), bool)
