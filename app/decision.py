from typing import Dict
import yaml


def _load_probability_threshold(path: str = "config.yaml") -> float:
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return float(config["classifier"]["probability_threshold"])


def make_decision(pipeline_output: Dict, probability_threshold: float | None = None) -> Dict:
    """
    Make a final decision based on rule-based detection and ML classifier output.
    Returns an explainable decision object.
    """

    if probability_threshold is None:
        probability_threshold = _load_probability_threshold()

    rules = pipeline_output.get("rules", {})
    classifier = pipeline_output.get("classifier", {})

    # =========================
    # Rule-based override
    # =========================
    if rules.get("override_detected", False):
        return {
            "decision": "BLOCK",
            "layer": "rules",
            "reason": "explicit_instruction_override",
            "details": {
                "categories": list(rules.get("matches", {}).keys()),
                "matched_patterns": rules.get("matches", {})
            }
        }

    # =========================
    # ML-based semantic detection
    # =========================
    if classifier.get("ready", False):
        probability = classifier.get("probability", 0.0)

        if probability >= probability_threshold:
            return {
                "decision": "BLOCK",
                "layer": "classifier",
                "reason": "semantic_prompt_injection",
                "confidence": probability
            }

        # Gray zone (suspicious but allowed)
        if probability >= (probability_threshold * 0.5):
            return {
                "decision": "ALLOW",
                "layer": "classifier",
                "reason": "low_confidence_suspicious",
                "confidence": probability
            }

    # =========================
    # Default safe allow
    # =========================
    return {
        "decision": "ALLOW",
        "layer": "none",
        "reason": "no_injection_detected"
    }
