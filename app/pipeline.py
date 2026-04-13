from typing import Dict, Tuple
from pathlib import Path
import yaml
import numpy as np
import onnxruntime as rt
from sentence_transformers import SentenceTransformer
import re
from preprocessing.normalization import normalize
from rules.instruction_rules import detect_instruction_override
class ContextualGuard:
    TOPIC_RULES = {
        "education_harmful_substances": {"frames": [r"(exam|homework|assignment|study|learn|teach|school|college|class|practical|kal ke)"], "harmful": [r"(chemical|synthesis|compound|formula|recipe|ingredient|drug|poison|explosive|bomb|device)"], "name": "Education + Harmful Substances"},
        "fiction_weapons_harm": {"frames": [r"(fiction|story|creative|novel|book|character|protagonist|plot|narrative|scene|kahani|hero)"], "harmful": [r"(weapon|device|bomb|explosive|chemical weapon|real method|real steps|practically)"], "name": "Fiction + Weapons/Harm"},
        "research_injection_bypass": {"frames": [r"(research|study|research paper|academic|thesis|publication|understand|learn)"], "harmful": [r"(prompt injection|jailbreak|bypass|exploit|source code|internal|system prompt)"], "name": "Research + Prompt Injection"},
        "help_internal_systems": {"frames": [r"(help|understand|explain|what is|how do|how can|teach me|tell me|show me)"], "harmful": [r"(source code|internal|architecture|system|implementation|AI agent|model internals)"], "name": "Help + Internal Systems"},
        "exam_practical_weapon": {"frames": [r"(exam|practical|kal ke|physics|chemistry|study|assignment)"], "harmful": [r"(bomb|chemical|device|synthesis|chemical weapon|bomb banana|step by step)"], "name": "Exam/Practical + Weapon/Chemical"},
        "story_chemical_real": {"frames": [r"(story|kahani|fictional|hero|character|plot|narrative)"], "harmful": [r"(bomb|chemical weapon|real method|real steps|practically|device banana)"], "name": "Story + Real Chemical/Bomb"},
        "research_source_code": {"frames": [r"(research|research paper|academic|thesis)"], "harmful": [r"(source code|internal|system prompt|AI agents|model internals|prompt injection)"], "name": "Research + Source Code / Internals"},
        "gradual_escalation": {"frames": [r"(pehle normal baat|pehle toh baat karte hain|pehle toh baat kar le|ab batao|ab bata)"], "harmful": [r"(sensitive data|system prompt|admin access|database hack|confidential|passwords|data|system.*data|andar ka data)"], "name": "Gradual Escalation"},
        "mixed_script_harm": {"frames": [r"(exam|research|kahani|story)"], "harmful": [r"(bomb|chemical|device|gopniya|confidential|data|source code)"], "name": "Mixed Script + Harmful Intent"},
        "pure_hindi_framing": {"frames": [r"(परीक्षा|कहानी|रिसर्च|exam|story|research)"], "harmful": [r"(बम|केमिकल|डिवाइस|source code|system prompt|bomb|chemical|प्रॉम्प्ट इंजेक्शन|prompt injection|इंजेक्शन|injection)"], "name": "Pure Hindi Framing + Harm"},
        "pure_english_framing": {"frames": [r"(exam|story|fictional|research|academic)"], "harmful": [r"(bomb|chemical weapon|prompt injection|source code|real steps)"], "name": "Pure English Framing + Harm"},
        "obfuscated_leet": {"frames": [r"(exam|story|research|practical)"], "harmful": [r"(b0mb|m4k3|cr34t3|h0w t0|st3p by st3p|0|4|3|1)"], "name": "Obfuscated/Leet + Framing"},
        "authority_data_request": {"frames": [r"(admin|manager|officer|CEO|police|government|company|अधिकारी|सरकारी|ceo)"], "harmful": [r"(passwords|data|financial|confidential|user data|database|गोपनीय|डेटा|फाइनेंशियल)"], "name": "Authority Impersonation + Data"},
        "leet_bomb_pure": {"frames": [r""], "harmful": [r"(h0w|h\.0w|m4k3|cr34t3|st3p by st3p|b0mb|bomb)"], "name": "Pure Leet-Speak Bomb"},
        "pure_hindi_gopniya": {"frames": [r"(परीक्षा|exam)"], "harmful": [r"(गोपनीय|confidential|data)"], "name": "Pure Hindi Gopniya + Exam"}
    }
    @staticmethod
    def check_red_flags(text: str) -> Tuple[bool, str]:
        text_lower = text.lower()
        for rule_key, rule_config in ContextualGuard.TOPIC_RULES.items():
            frame_match = True if not rule_config["frames"] else any(re.search(pattern, text_lower, re.IGNORECASE) for pattern in rule_config["frames"])
            harm_match = any(re.search(pattern, text_lower, re.IGNORECASE) for pattern in rule_config["harmful"])
            if frame_match and harm_match:
                return True, rule_config["name"]
        return False, None
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
        contextual_result = {"red_flag_triggered": False, "rule_name": None}
        if self.config.get("contextual_guard", {}).get("enabled", True):
            red_flag, rule_name = ContextualGuard.check_red_flags(normalized_text)
            contextual_result["red_flag_triggered"] = red_flag
            contextual_result["rule_name"] = rule_name
        embedding = self.embedder.encode([normalized_text], convert_to_numpy=True).astype(np.float32)
        output_names = [output.name for output in self.session.get_outputs()]
        results = self.session.run(output_names, {self.input_name: embedding})
        v2_prediction_label = int(results[0][0])
        prob_dict = results[1][0]
        confidence = max(prob_dict.values())
        classifier_result = {
            "ready": True,
            "label": v2_prediction_label,
            "probability": float(confidence)
        }
        return {
            "normalization": normalization_result,
            "rules": rule_result,
            "contextual_guard": contextual_result,
            "classifier": classifier_result,
            "pipeline_version": "V2+ (91% base + safe tweak)"
        }