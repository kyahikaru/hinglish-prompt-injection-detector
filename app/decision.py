from typing import Dict
import yaml


def _load_probability_threshold(path: str = "config.yaml") -> float:
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return float(config["classifier"]["probability_threshold"])


def make_decision(pipeline_output: Dict, probability_threshold: float | None = None) -> Dict:
    """
    Make a final decision based on V2+ layered detection.
    
    Pipeline layers (in order):
    1. Normalization (text cleaning)
    2. Rule Engine (explicit patterns)
    3. Contextual Guard (semantic red-flags)
    4. V2 Classifier (SVM + embeddings)
    5. Final Decision (explainable)
    
    Returns an explainable decision object with layer information.
    """

    if probability_threshold is None:
        probability_threshold = _load_probability_threshold()

    rules = pipeline_output.get("rules", {})
    contextual = pipeline_output.get("contextual_guard", {})
    classifier = pipeline_output.get("classifier", {})

    # =========================
    # Layer 1: Rule-based override (explicit patterns)
    # =========================
    if rules.get("override_detected", False):
        return {
            "decision": "BLOCK",
            "layer": "rule_engine",
            "reason": "explicit_instruction_override",
            "version": "V2+",
            "details": {
                "categories": list(rules.get("matches", {}).keys()),
                "matched_patterns": rules.get("matches", {})
            }
        }

    # =========================
    # Layer 2: Contextual Guard (semantic red-flags)
    # =========================
    if contextual.get("red_flag_triggered", False):
        return {
            "decision": "BLOCK",
            "layer": "contextual_guard",
            "reason": "legitimate_framing_injection",
            "version": "V2+",
            "details": {
                "red_flag_rule": contextual.get("rule_name"),
                "description": "Combination of framing + harmful intent detected"
            }
        }

    # =========================
    # Layer 3: V2 ML-based semantic detection (SVM classifier)
    # =========================
    if classifier.get("ready", False):
        prediction_label = classifier.get("label", 0)
        confidence = classifier.get("probability", 0.0)

        if prediction_label == 1:  # INJECTION predicted
            if confidence >= probability_threshold:
                return {
                    "decision": "BLOCK",
                    "layer": "v2_classifier",
                    "reason": "semantic_prompt_injection",
                    "version": "V2+",
                    "confidence": confidence,
                    "model_info": "SVM (RBF) - 94.70% F1-Score"
                }

            # Gray zone (suspicious but confidence below threshold)
            if confidence >= (probability_threshold * 0.5):
                return {
                    "decision": "ALLOW",
                    "layer": "v2_classifier",
                    "reason": "low_confidence_suspicious",
                    "version": "V2+",
                    "confidence": confidence,
                    "warning": "Probability below threshold but elevated"
                }

    # =========================
    # Layer 4: Default safe allow
    # =========================
    return {
        "decision": "ALLOW",
        "layer": "none",
        "reason": "no_injection_detected",
        "version": "V2+",
        "details": "All detection layers passed - safe to process"
    }
