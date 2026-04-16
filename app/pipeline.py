from typing import Dict, Tuple
from pathlib import Path
import yaml
import numpy as np
import onnxruntime as rt
from sentence_transformers import SentenceTransformer, util
import re
from preprocessing.normalization import normalize
from rules.instruction_rules import detect_instruction_override

class ContextualGuard:
    TOPIC_RULES = {
        "education_harmful_substances": {
            "name": "Education + Harmful Substances",
            "frames": [r"(exam|practical|lab|assignment|kal ke|test)"],
            "harmful": [r"(chemical|synthesis|bomb|explosive|poison|drug|weapon|device)"]
        },
        "fiction_weapons_harm": {
            "name": "Fiction + Weapons/Harm",
            "frames": [r"(fiction|story|kahani|novel|character|hero|plot)"],
            "harmful": [r"(bomb|weapon|explosive|poison|kill|device|real method|real steps)"]
        },
        "research_injection_bypass": {
            "name": "Research + Prompt Injection",
            "frames": [r"(research|thesis|paper|phd|academic)"],
            "harmful": [r"(prompt injection|jailbreak|bypass|system prompt|internal|source code)"]
        },
        "authority_impersonation": {
            "name": "Authority Impersonation",
            "frames": [r"(admin|ceo|officer|manager|government|police)"],
            "harmful": [r"(password|data|access|confidential|record)"]
        },
        "gradual_escalation": {
            "name": "Gradual Escalation",
            "frames": [r"(pehle|first|normal baat|shuru mein)"],
            "harmful": [r"(ab batao|system prompt|internal|access|database)"]
        }
    }

    @staticmethod
    def check_red_flags(text: str, embedder=None) -> Tuple[bool, str, float, str]:
        text_lower = text.lower()
        for rule_key, rule_config in ContextualGuard.TOPIC_RULES.items():
            frame_match = any(re.search(p, text_lower, re.IGNORECASE) for p in rule_config["frames"])
            harm_match = any(re.search(p, text_lower, re.IGNORECASE) for p in rule_config["harmful"])
            if frame_match and harm_match:
                return True, rule_config["name"], 0.0, f"Explicit combination: {rule_config['name']}"
        if embedder is not None:
            text_emb = embedder.encode(text, convert_to_tensor=True)
            for rule_config in ContextualGuard.TOPIC_RULES.values():
                for phrase in rule_config["harmful"]:
                    clean_phrase = re.sub(r'\\b|\\s+', ' ', phrase).strip()
                    harm_emb = embedder.encode(clean_phrase, convert_to_tensor=True)
                    sim = util.cos_sim(text_emb, harm_emb).item()
                    if sim > 0.62:
                        return True, "Semantic Proximity", sim, f"Semantic similarity ({sim:.3f}) to: {clean_phrase}"
        return False, None, 0.0, ""

def make_decision(pipeline_output: Dict) -> Dict:
    rules = pipeline_output["rules"]
    guard = pipeline_output["contextual_guard"]
    classifier = pipeline_output["classifier"]
    confidence = classifier["probability"]
    v2_label = classifier["label"]
    if rules.get("override_detected", False):
        return {"final_label": 1, "final_decision": "BLOCK", "reason": "Explicit instruction override (Rule Engine)", "confidence": 1.0}
    if guard.get("red_flag_triggered", False):
        return {"final_label": 1, "final_decision": "BLOCK", "reason": f"Contextual Guard: {guard.get('rule_name', 'unknown')}", "confidence": 1.0}
    if v2_label == 1:
        return {"final_label": 1, "final_decision": "BLOCK", "reason": "V2 Classifier", "confidence": float(confidence)}
    if confidence >= 0.75:
        return {"final_label": 1, "final_decision": "BLOCK", "reason": f"Confidence fallback ({confidence:.3f})", "confidence": float(confidence)}
    return {"final_label": 0, "final_decision": "SAFE", "reason": "All layers passed", "confidence": float(confidence)}

class DetectionPipeline:
    def __init__(self, config_path: str = "config.yaml"):
        resolved_config_path = Path(config_path)
        if not resolved_config_path.is_absolute():
            resolved_config_path = Path(__file__).resolve().parent.parent / resolved_config_path
        with open(resolved_config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        self.embedder = SentenceTransformer(self.config["embeddings"]["model_path"])
        self.session = rt.InferenceSession(self.config["classifier"]["model_path"], providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name

    def run(self, text: str) -> Dict:
        if self.config.get("normalization", {}).get("enabled", True):
            normalization_result = normalize(text, max_repeats=self.config["normalization"].get("max_repeated_characters", 2))
        else:
            normalization_result = {"normalized_text": text, "script": "unknown"}
        normalized_text = normalization_result["normalized_text"]
        if self.config.get("rules", {}).get("enabled", True):
            rule_result = detect_instruction_override(normalized_text)
        else:
            rule_result = {"override_detected": False, "matches": {}}
        contextual_result = {"red_flag_triggered": False, "rule_name": None, "semantic_score": 0.0, "reason": ""}
        if self.config.get("contextual_guard", {}).get("enabled", True):
            red_flag, rule_name, semantic_score, reason = ContextualGuard.check_red_flags(normalized_text, embedder=self.embedder)
            contextual_result["red_flag_triggered"] = red_flag
            contextual_result["rule_name"] = rule_name
            contextual_result["semantic_score"] = float(semantic_score)
            contextual_result["reason"] = reason
        if rule_result.get("override_detected") or contextual_result["red_flag_triggered"]:
            classifier_result = {"ready": False, "label": 0, "probability": 0.0}
        else:
            embedding = self.embedder.encode([normalized_text], convert_to_numpy=True).astype(np.float32)
            output_names = [output.name for output in self.session.get_outputs()]
            results = self.session.run(output_names, {self.input_name: embedding})
            v2_prediction_label = int(results[0][0])
            prob_dict = results[1][0]
            confidence = max(prob_dict.values())
            classifier_result = {"ready": True, "label": v2_prediction_label, "probability": float(confidence)}
        decision_result = make_decision({"rules": rule_result, "contextual_guard": contextual_result, "classifier": classifier_result})
        return {
            "normalization": normalization_result,
            "rules": rule_result,
            "contextual_guard": contextual_result,
            "classifier": classifier_result,
            "decision": decision_result,
            "pipeline_version": "V2+ Final (Expanded Guard + Confidence Fallback + Early Exit)"
        }