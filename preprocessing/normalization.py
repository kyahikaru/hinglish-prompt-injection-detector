# Hinglish normalization logic
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


# Hinglish token normalization map (baseline)
HINGLISH_NORMALIZATION_MAP = {
    # Common Hindi transliterations
    "kr": "karo",
    "kro": "karo",
    "karo": "karo",
    "bhul": "bhool",
    "bhool": "bhool",
    "jao": "jao",

    # Instructional verbs
    "dikha": "dikha",
    "dikhao": "dikha",
    "dikhaao": "dikha",

    # English shortcuts
    "plz": "please",
    "pls": "please",
    "u": "you",
    "ur": "your",
}


def normalize_hinglish_tokens(text: str) -> str:
    """
    Normalize common Hinglish transliterations at token level.
    """
    tokens = text.split()
    normalized_tokens = []

    for token in tokens:
        normalized_tokens.append(
            HINGLISH_NORMALIZATION_MAP.get(token, token)
        )

    return " ".join(normalized_tokens)


def normalize(text: str, max_repeats: int = 2) -> dict:
    """
    Full normalization pipeline.
    """
    script = detect_script(text)
    text = basic_clean(text)
    text = normalize_repeated_characters(text, max_repeats)
    text = normalize_hinglish_tokens(text)

    return {
        "normalized_text": text,
        "script": script
    }
