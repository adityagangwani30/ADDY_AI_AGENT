import os
import tempfile

from services import ocr_service


def test_ocr_image_no_tesseract():
    # If pytesseract not installed, the function should return empty string gracefully
    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    try:
        text = ocr_service.ocr_image(path)
        assert isinstance(text, str)
    finally:
        try:
            os.remove(path)
        except Exception:
            pass
