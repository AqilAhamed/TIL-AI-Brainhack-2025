from typing import Any
from PIL import Image
import pytesseract
from io import BytesIO


class OCRManager:

    def __init__(self):
        # You can configure Tesseract path here if needed
        # Example: pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
        pass

    def ocr(self, image: bytes) -> str:
        """Performs OCR on an image of a document.

        Args:
            image: The image file in bytes.

        Returns:
            A string containing the text extracted from the image.
        """

        # Load image from bytes
        img = Image.open(BytesIO(image)).convert("RGB")

        # Perform OCR using pytesseract
        text = pytesseract.image_to_string(img)

        return text.strip()
