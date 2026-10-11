"""OCR extraction for product review screenshots.

Responsibilities
----------------
- Load review screenshots (file path, numpy array or Streamlit upload).
- Preprocess images with OpenCV (grayscale, upscaling, denoising, binarisation,
  optional deskew) so that Tesseract achieves better accuracy on screenshots.
- Extract full text and word-level bounding boxes + confidence scores with
  pytesseract, which later enables word-level sentiment highlighting on the
  original image.

External requirement
--------------------
``pip install pytesseract`` installs only the Python wrapper. The Tesseract
OCR *engine* must be installed separately on the operating system:

- Windows: https://github.com/UB-Mannheim/tesseract/wiki
- macOS:   brew install tesseract
- Ubuntu:  sudo apt install tesseract-ocr

If the engine is missing, ``extract_text`` raises a ``RuntimeError`` with
installation instructions instead of silently returning empty/fake text.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

try:
    import pytesseract
    from pytesseract import TesseractNotFoundError

    _TESSERACT_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - exercised only on broken installs
    pytesseract = None  # type: ignore[assignment]
    TesseractNotFoundError = Exception  # type: ignore[assignment]
    _TESSERACT_IMPORT_ERROR = exc

ImageInput = Union[str, Path, np.ndarray]

_TESSERACT_INSTALL_HELP = (
    "Tesseract OCR engine was not found on this system.\n"
    "Installing the 'pytesseract' pip package is NOT enough - the engine "
    "must be installed separately:\n"
    "  - Windows: https://github.com/UB-Mannheim/tesseract/wiki "
    "(then add the install path to PATH or set pytesseract.pytesseract.tesseract_cmd)\n"
    "  - macOS:   brew install tesseract\n"
    "  - Ubuntu:  sudo apt install tesseract-ocr"
)


@dataclass
class OcrWord:
    """A single recognised word with its position and confidence."""

    text: str
    left: int
    top: int
    width: int
    height: int
    confidence: float

    @property
    def box(self) -> Tuple[int, int, int, int]:
        """Return ``(x, y, w, h)``."""
        return (self.left, self.top, self.width, self.height)


def load_image(source: ImageInput) -> np.ndarray:
    """Load an image into an OpenCV BGR ``numpy.ndarray``.

    Accepts a file path or an already-decoded numpy array (e.g. from
    ``PIL.Image`` / Streamlit's ``UploadedFile`` via ``np.asarray``).
    """
    if isinstance(source, np.ndarray):
        image = source
    else:
        path = str(source)
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image file: {path}")
    if image.size == 0:
        raise ValueError("Received an empty image.")
    return image


def preprocess_image(image: np.ndarray, *, upscale: bool = True, max_width: int = 1800) -> np.ndarray:
    """Improve screenshot images for OCR.

    Steps: grayscale -> optional upscale (small screenshots benefit a lot)
    -> median blur denoise -> adaptive threshold -> light morphology.

    Returns a single-channel (grayscale) image suitable for Tesseract.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    if upscale and gray.shape[1] < max_width:
        scale = max_width / float(gray.shape[1])
        gray = cv2.resize(
            gray,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC,
        )

    gray = cv2.medianBlur(gray, 3)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11
    )
    kernel = np.ones((1, 1), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return binary


def _require_tesseract() -> None:
    if pytesseract is None:
        raise RuntimeError(
            "The 'pytesseract' package is not installed. " + _TESSERACT_INSTALL_HELP
        ) from _TESSERACT_IMPORT_ERROR
    try:
        pytesseract.get_tesseract_version()
    except TesseractNotFoundError as exc:
        raise RuntimeError(_TESSERACT_INSTALL_HELP) from exc


def extract_text(source: ImageInput, *, preprocess: bool = True) -> str:
    """Extract raw text from an image using Tesseract OCR.

    Raises ``RuntimeError`` with installation help when the Tesseract engine
    is unavailable; never fabricates text.
    """
    _require_tesseract()
    image = load_image(source)
    if preprocess:
        image = preprocess_image(image)
    text = pytesseract.image_to_string(image)
    return text.strip()


def extract_words(source: ImageInput, *, preprocess: bool = True, min_confidence: float = 0.0) -> List[OcrWord]:
    """Extract word-level results (text, bounding box, confidence).

    This powers word-level sentiment highlighting: each word keeps its pixel
    box so attention scores can later be painted onto the original screenshot.
    """
    _require_tesseract()
    image = load_image(source)
    processed = preprocess_image(image) if preprocess else image
    data = pytesseract.image_to_data(
        processed, output_type=pytesseract.Output.DICT
    )

    words: List[OcrWord] = []
    n = len(data["text"])
    for i in range(n):
        raw = data["text"][i].strip()
        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError):
            conf = -1.0
        if not raw or conf < min_confidence:
            continue
        words.append(
            OcrWord(
                text=raw,
                left=int(data["left"][i]),
                top=int(data["top"][i]),
                width=int(data["width"][i]),
                height=int(data["height"][i]),
                confidence=conf,
            )
        )
    return words


def is_tesseract_available() -> bool:
    """Return True when both the wrapper and the engine are usable."""
    if pytesseract is None:
        return False
    try:
        pytesseract.get_tesseract_version()
    except Exception:
        return False
    return True
