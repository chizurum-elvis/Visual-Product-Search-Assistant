import pytesseract
from PIL import Image
import logging
from tenacity import retry, stop_after_attempt
import numpy as np

logger = logging.getLogger("OCR")


@retry(stop=stop_after_attempt(2))
def extract_text_from_image(image_path: str) -> str:
    """Robust text extraction with preprocessing"""
    try:
        with Image.open(image_path) as img:
            # Preprocess for better OCR
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Contrast enhancement
            img_array = np.array(img)
            img_array = (img_array * 1.2).clip(0, 255).astype('uint8')
            processed_img = Image.fromarray(img_array)

            text = pytesseract.image_to_string(processed_img)
            logger.debug(f"Extracted {len(text)} characters from image")
            return text.strip() or "No text detected"
    except Exception as e:
        logger.error(f"OCR failed for {image_path}", exc_info=True)
        return ""