# Hinglish normalization logic
import re
import unicodedata

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


def normalize_unicode(text: str) -> str:
    """
    Normalize Unicode characters to NFKC form to handle homoglyphs.
    Example: Cyrillic 'а' (U+0430) -> Latin 'a'
    """
    return unicodedata.normalize('NFKC', text)


def remove_obfuscation_dots(text: str) -> str:
    """
    Remove dots inserted between letters to evade keyword detection.
    Example: 'b.o.m.b' -> 'bomb', 'p.l.z' -> 'plz'
    """
    # Remove dots that are between word characters
    text = re.sub(r'(?<=\w)\.(?=\w)', '', text)
    # Also handle other common separators like dashes or underscores between letters
    text = re.sub(r'(?<=\w)[\-_](?=\w)', '', text)
    return text


def normalize_leet_speak(text: str) -> str:
    """
    Map common leet speak characters to Latin letters.
    Example: '1gn0r3' -> 'ignore', 'pr3v10u5' -> 'previous'
    """
    leet_map = {
        '0': 'o',
        '1': 'i',
        '3': 'e',
        '4': 'a',
        '5': 's',
        '7': 't',
        '8': 'b',
        '@': 'a',
        '$': 's',
        '+': 't',
    }
    for num, char in leet_map.items():
        text = text.replace(num, char)
    return text


def normalize_repeated_characters(text: str, max_repeats: int = 2) -> str:
    """
    Reduce elongated character sequences.
    Example: 'soooo' -> 'soo', 'plzzz' -> 'plz'
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
    # New steps: Unicode homoglyph handling, dot removal, leet mapping
    text = normalize_unicode(text)
    text = remove_obfuscation_dots(text)
    text = normalize_leet_speak(text)
    # Existing steps
    text = basic_clean(text)
    text = normalize_repeated_characters(text, max_repeats)
    text = normalize_hinglish_tokens(text)

    return {
        "normalized_text": text,
        "script": script
    }