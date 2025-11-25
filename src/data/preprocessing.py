import re

def clean_text(text: str) -> str:
    """
    Clean raw OCR text by removing extra whitespace, fixing line breaks,
    and normalizing common characters.
    """
    if not text:
        return ""

    # Normalize unicode characters
    text = text.replace("\u00A0", " ")

    # Remove multiple spaces
    text = re.sub(r"\s+", " ", text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


def extract_basic_features(text: str) -> dict:
    """
    Extract simple numeric/text features from OCR text.
    These will be used by baseline ML models.
    """
    clean = clean_text(text)

    # Count how many digits (useful for totals, amounts)
    digit_count = sum(char.isdigit() for char in clean)

    # Count lines
    line_count = len(text.split("\n"))

    # Find Euro amounts like 1.234,56 or 1234.56
    euro_amounts = re.findall(r"\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})", clean)

    # Find dates like 12.03.2022 or 2022-03-12
    dates = re.findall(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b", clean)

    return {
        "clean_text": clean,
        "digit_count": digit_count,
        "line_count": line_count,
        "euro_amounts": euro_amounts,
        "dates": dates,
    }
