# Final decision logic will go here
from typing import Dict


def make_decision(
    rule_result: Dict,
    classifier_result: Dict,
    probability_threshold: float
) -> Dict:
    """
    Make final allow/block decision based on rule and classifier outputs.
    """

    # Rule-based override has highest priority
    if rule_result.get("override_detected", False):
        return {
            "decision": "block",
            "reason": "explicit_instruction_override"
        }

    # Semantic classifier decision
    if classifier_result.get("ready", False):
        if classifier_result.get("probability", 0.0) >= probability_threshold:
            return {
                "decision": "block",
                "reason": "semantic_prompt_injection"
            }

    # Default allow
    return {
        "decision": "allow",
        "reason": "no_injection_detected"
    }
