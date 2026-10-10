"""Optical Character Recognition (OCR) utilities with EasyOCR singleton and disk caching."""

import io
import logging
import os
from pathlib import Path
import threading
from typing import Any, Optional, Tuple, Union

from docurag.cache import disk_cache
from docurag.config import get_settings

logger = logging.getLogger(__name__)

_READER_LOCK = threading.Lock()
_EASYOCR_READER: Optional[Any] = None


def _is_fakes() -> bool:
    if os.getenv("DOCURAG_FAKES") in ("1", "true", "True"):
        return True
    try:
        return get_settings().DOCURAG_FAKES
    except Exception:
        return False


def get_ocr_reader() -> Any:
    """Retrieve or lazily initialize the thread-safe singleton EasyOCR Reader.

    Detects CPU vs CUDA automatically using torch.cuda.is_available().
    Under DOCURAG_FAKES=1 or if easyocr is unavailable, returns None or a mock without failing.
    """
    global _EASYOCR_READER
    if _EASYOCR_READER is None:
        with _READER_LOCK:
            if _EASYOCR_READER is None:
                if _is_fakes():
                    return None
                try:
                    import easyocr
                    import torch
                    use_gpu = torch.cuda.is_available()
                    _EASYOCR_READER = easyocr.Reader(["en"], gpu=use_gpu, verbose=False)
                except Exception as exc:
                    logger.warning("Failed to initialize EasyOCR reader: %s", exc)
                    return None
    return _EASYOCR_READER


@disk_cache(namespace="ocr")
def _ocr_image_bytes(image_bytes: bytes) -> Tuple[str, float]:
    """Internal cached OCR runner on normalized image bytes."""
    if _is_fakes():
        return ("Simulated OCR extracted text content.", 0.95)

    reader = get_ocr_reader()
    if reader is None:
        return ("", 0.0)

    try:
        import numpy as np
        from PIL import Image

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_np = np.array(image)

        results = reader.readtext(img_np)
        if not results:
            return ("", 0.0)

        # Sort top-to-bottom, left-to-right based on top-left y, then x
        sorted_results = sorted(
            results,
            key=lambda item: (item[0][0][1] if item[0] else 0, item[0][0][0] if item[0] else 0),
        )

        texts = [str(item[1]).strip() for item in sorted_results if item[1]]
        confs = [float(item[2]) for item in sorted_results if len(item) > 2]

        extracted_text = "\n\n".join(texts)
        mean_conf = sum(confs) / len(confs) if confs else 0.0

        return (extracted_text, float(mean_conf))
    except Exception as exc:
        logger.warning("OCR processing error: %s", exc)
        return ("", 0.0)


def ocr_page(image: Any) -> Tuple[str, float]:
    """Perform OCR on a page image or file path with disk caching and confidence scoring.

    Args:
        image: File path (Path/str), image bytes, or PIL Image.

    Returns:
        Tuple of (extracted_text, mean_confidence_score).
    """
    if _is_fakes():
        return ("Simulated OCR extracted text content.", 0.95)

    try:
        image_bytes: bytes
        if isinstance(image, (str, Path)):
            p = Path(image)
            if not p.exists():
                return ("", 0.0)
            image_bytes = p.read_bytes()
        elif isinstance(image, bytes):
            image_bytes = image
        elif hasattr(image, "save"):
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            image_bytes = buf.getvalue()
        else:
            return ("", 0.0)

        res = _ocr_image_bytes(image_bytes)
        return (str(res[0]), float(res[1]))
    except Exception as exc:
        logger.warning("ocr_page failed with error: %s", exc)
        return ("", 0.0)
