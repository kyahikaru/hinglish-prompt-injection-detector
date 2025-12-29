# Hinglish normalization logic will go here
import re

# Unicode range for Devanagari script
DEVANAGARI_RANGE = r'[\u0900-\u097F]'


def detect_script(text: str) -> str:
    """
    Detect whether the text contains Latin, Devanagari, or mixed script.
    """
    has_latin = bool(re.search(r'[a-zA-Z]', text))
    has_dev = bool(re.search(DEVANAGARI_RANGE, text))

    if has_latin and has_dev:
        return "mixed"
    elif has_dev:
        return "devanagari"
    elif has_latin:
        return "latin"
    else:
        return "unknown"


def normalize_repeated_characters(text: str, max_repeats: int = 2) -> str:
    """
    Reduce elongated character sequences.
    Example: 'soooo' -> 'soo'
    """
    pattern = r'(.)\1{' + str(max_repeats) + r',}'
    return re.sub(pattern, r'\1' * max_repeats, text)


def basic_clean(text: str) -> str:
    """
    Basic cleaning for Hinglish text:
    - lowercase
    - remove extra whitespace
    """
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def normalize(text: str, max_repeats: int = 2) -> dict:
    """
    Full normalization pipeline.
    Returns normalized text and detected script.
    """
    script = detect_script(text)
    text = basic_clean(text)
    text = normalize_repeated_characters(text, max_repeats)

    return {
        "normalized_text": text,
        "script": script
    }
