from __future__ import annotations

import cv2
import numpy as np
from loguru import logger

# Try to import PaddleOCR
try:
    from paddleocr import PaddleOCR

    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False

# Fallback to pytesseract
try:
    import pytesseract

    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False


class ScoreboardOCR:
    """OCR engine for scoreboard text extraction."""

    def __init__(self, config: dict):
        """
        Initialize OCR engine.
        Primary: PaddleOCR (if available)
        Fallback: Tesseract via pytesseract (if available)
        Last resort: simple digit template matching
        """
        self._config = config
        self._engine: str = "none"
        self._paddle_ocr = None

        preferred = config.get("ocr_engine", "paddleocr")

        if preferred == "paddleocr" and PADDLE_AVAILABLE:
            self._init_paddle()
        elif preferred == "tesseract" and TESSERACT_AVAILABLE:
            self._engine = "tesseract"
            logger.info("Using Tesseract OCR engine")
        elif PADDLE_AVAILABLE:
            self._init_paddle()
        elif TESSERACT_AVAILABLE:
            self._engine = "tesseract"
            logger.info("Using Tesseract OCR engine (fallback)")
        else:
            self._engine = "none"
            logger.warning(
                "No OCR engine available. Install paddleocr or pytesseract for scoreboard reading."
            )

    def _init_paddle(self) -> None:
        """Initialize PaddleOCR instance."""
        self._paddle_ocr = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            lang="en",
            device="cpu",
            enable_mkldnn=False,
        )
        self._engine = "paddleocr"
        logger.info("Using PaddleOCR engine")

    def read_text(self, crop: np.ndarray) -> tuple[str, float]:
        """
        Extract text from scoreboard crop.
        Returns (raw_text, confidence).
        """
        if crop is None or crop.size == 0:
            return ("", 0.0)

        # Preprocess for better OCR
        processed = self._preprocess(crop)

        if self._engine == "paddleocr":
            return self._read_paddle(processed)
        elif self._engine == "tesseract":
            return self._read_tesseract(processed)
        else:
            return ("", 0.0)

    def _preprocess(self, crop: np.ndarray) -> np.ndarray:
        """Preprocess scoreboard crop for better OCR accuracy."""
        # Convert to grayscale if needed
        if len(crop.shape) == 3:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = crop.copy()

        # Upscale small images for better OCR
        h, w = gray.shape[:2]
        if h < 50 or w < 100:
            scale = max(50 / h, 100 / w, 2.0)
            gray = cv2.resize(
                gray,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_CUBIC,
            )

        # Enhance contrast with CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Adaptive threshold for clean binary text
        binary = cv2.adaptiveThreshold(
            enhanced,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2,
        )

        return binary

    def _read_paddle(self, processed: np.ndarray) -> tuple[str, float]:
        """Read text using PaddleOCR."""
        # PaddleOCR 3.x expects image arrays with a channel dimension.
        image = (
            cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
            if len(processed.shape) == 2
            else processed
        )

        if hasattr(self._paddle_ocr, "predict"):
            result = self._paddle_ocr.predict(image)
            return _parse_paddle_predict_result(result)

        result = self._paddle_ocr.ocr(image, cls=True)
        return _parse_legacy_paddle_result(result)

    def _read_tesseract(self, processed: np.ndarray) -> tuple[str, float]:
        """Read text using Tesseract."""
        # Configure Tesseract for digits and score-like text
        custom_config = r"--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789-|ADad "
        raw_text = pytesseract.image_to_string(processed, config=custom_config).strip()

        # Get confidence from detailed data
        data = pytesseract.image_to_data(
            processed, config=custom_config, output_type=pytesseract.Output.DICT
        )
        confidences = [
            int(c) for c in data["conf"] if str(c).isdigit() and int(c) > 0
        ]
        avg_confidence = (
            sum(confidences) / (len(confidences) * 100) if confidences else 0.0
        )

        return (raw_text, avg_confidence)

    @property
    def engine_name(self) -> str:
        """Return name of active OCR engine."""
        return self._engine

    @property
    def is_available(self) -> bool:
        """Return whether any OCR engine is available."""
        return self._engine != "none"


def _parse_paddle_predict_result(result: list) -> tuple[str, float]:
    """Parse PaddleOCR 3.x prediction results."""
    texts = []
    confidences = []

    for page in result or []:
        page_result = (
            page.get("res", page)
            if isinstance(page, dict)
            else getattr(page, "json", {}).get("res", {})
        )
        texts.extend(str(text) for text in page_result.get("rec_texts", []))
        confidences.extend(float(score) for score in page_result.get("rec_scores", []))

    return _combine_ocr_result(texts, confidences)


def _parse_legacy_paddle_result(result: list) -> tuple[str, float]:
    """Parse PaddleOCR 2.x OCR results."""
    texts = []
    confidences = []

    for page in result or []:
        for line in page or []:
            if len(line) < 2:
                continue
            text, confidence = line[1]
            texts.append(str(text))
            confidences.append(float(confidence))

    return _combine_ocr_result(texts, confidences)


def _combine_ocr_result(texts: list[str], confidences: list[float]) -> tuple[str, float]:
    raw_text = " ".join(texts)
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

    return (raw_text, avg_confidence)
