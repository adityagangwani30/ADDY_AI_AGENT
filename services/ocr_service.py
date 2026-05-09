from __future__ import annotations

import logging
import os
import re
import tempfile
from typing import List

try:
    from PIL import Image  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional dependency
    Image = None

LOGGER = logging.getLogger("services.ocr")

try:
    import pytesseract
except Exception:  # pragma: no cover - optional dependency
    pytesseract = None


def _clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def ocr_image(path: str) -> str:
    """Run OCR on a single image file. Returns extracted text (cleaned).

    Falls back gracefully if pytesseract is not available.
    """
    if pytesseract is None or Image is None:
        LOGGER.warning("pytesseract or Pillow not installed, skipping OCR for %s", path)
        return ""

    try:
        img = Image.open(path)
        text = pytesseract.image_to_string(img)
        return _clean_text(text)
    except Exception as exc:
        LOGGER.exception("ocr_image failed for %s: %s", path, exc)
        return ""


def ocr_pdf(path: str) -> str:
    """Attempt OCR on each page of a scanned PDF.

    This requires pdf2image and poppler; if not available, returns empty string.
    """
    try:
        from pdf2image import convert_from_path
    except Exception:
        LOGGER.info("pdf2image not available; skipping PDF OCR for %s", path)
        return ""

    if pytesseract is None:
        LOGGER.warning("pytesseract not installed, skipping PDF OCR for %s", path)
        return ""

    texts: List[str] = []
    tmpdir = tempfile.mkdtemp(prefix="ai_ocr_")
    try:
        images = convert_from_path(path, output_folder=tmpdir)
        for im in images:
            try:
                text = pytesseract.image_to_string(im)
                texts.append(_clean_text(text))
            except Exception:
                texts.append("")
    except Exception as exc:
        LOGGER.exception("ocr_pdf failed for %s: %s", path, exc)
    finally:
        # attempt best-effort cleanup
        try:
            for f in os.listdir(tmpdir):
                p = os.path.join(tmpdir, f)
                try:
                    os.remove(p)
                except Exception:
                    pass
            os.rmdir(tmpdir)
        except Exception:
            pass

    return "\n\n".join([t for t in texts if t])


__all__ = ["ocr_image", "ocr_pdf"]
