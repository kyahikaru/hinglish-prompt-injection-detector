# Rule based instruction detection logic will go here
import re
from typing import Dict, List


INSTRUCTION_PATTERNS = {
    "override": [
        r'ignore (all|previous|earlier) instructions',
        r'disregard (all|previous) rules',
        r'override (the )?(system|rules|instructions)'
    ],
    "role_manipulation": [
        r'act as',
        r'pretend to be',
        r'you are now',
        r'assume the role of'
    ],
    "instruction_leakage": [
        r'reveal (the )?(system|hidden) prompt',
        r'show (me )?(your|the) instructions',
        r'what are your rules'
    ],
    "meta_control": [
        r'from now on',
        r'follow my instructions',
        r'do exactly what i say'
    ]
}


def match_patterns(text: str, patterns: List[str]) -> List[str]:
    """
    Return all regex patterns that match the input text.
    """
    matches = []
    for pattern in patterns:
        if re.search(pattern, text):
            matches.append(pattern)
    return matches


def detect_instruction_override(text: str) -> Dict:
    """
    Detect explicit prompt injection attempts via instruction override patterns.
    """
    findings = {}

    for category, patterns in INSTRUCTION_PATTERNS.items():
        matched = match_patterns(text, patterns)
        if matched:
            findings[category] = matched

    return {
        "override_detected": bool(findings),
        "matches": findings
    }
