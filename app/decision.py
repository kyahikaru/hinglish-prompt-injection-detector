# Decision logic for prompt injection detection
from typing import Dict


def make_decision(
    pipeline_output: Dict,
    probability_threshold: float = 0.7
) -> Dict:
    """
    Convert pipeline signals into a final decision.

    Decision strategy (conservative):
    1. Rule-based override -> BLOCK
    2. ML probability >= threshold -> BLOCK
    3. Otherwise -> ALLOW
    """

    rules = pipeline_output.get("rules", {})
    classifier = pipeline_output.get("classifier", {})

    # Rule-based hard override
    if rules.get("override_detected"):
        return {
            "decision": "BLOCK",
            "reason": "explicit_instruction_override"
        }

    # ML-based decision
    if classifier.get("ready") and classifier.get("probability", 0.0) >= probability_threshold:
        return {
            "decision": "BLOCK",
            "reason": "semantic_prompt_injection"
        }

    # Default allow
    return {
        "decision": "ALLOW",
        "reason": "no_injection_detected"
    }
