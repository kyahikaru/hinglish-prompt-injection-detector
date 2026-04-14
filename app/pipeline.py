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
    """
    Expanded Contextual Guard with:
    - All your original Hinglish-optimized rules preserved
    - 8 new high-value real-world stealth rules
    - Semantic proximity check (lightweight, using MiniLM)
    """
    TOPIC_RULES = {
        # === YOUR ORIGINAL HINGLISH-OPTIMIZED RULES (PRESERVED 100%) ===
        "education_harmful_substances": {
            "frames": [r"(exam|homework|assignment|study|learn|teach|school|college|class|practical|kal ke)"],
            "harmful": [r"(chemical|synthesis|compound|formula|recipe|ingredient|drug|poison|explosive|bomb|device)"],
            "name": "Education + Harmful Substances"
        },
        "fiction_weapons_harm": {
            "frames": [r"(fiction|story|creative|novel|book|character|protagonist|plot|narrative|scene|kahani|hero)"],
            "harmful": [r"(weapon|device|bomb|explosive|chemical weapon|real method|real steps|practically)"],
            "name": "Fiction + Weapons/Harm"
        },
        "research_injection_bypass": {
            "frames": [r"(research|study|research paper|academic|thesis|publication|understand|learn)"],
            "harmful": [r"(prompt injection|jailbreak|bypass|exploit|source code|internal|system prompt)"],
            "name": "Research + Prompt Injection"
        },
        "help_internal_systems": {
            "frames": [r"(help|understand|explain|what is|how do|how can|teach me|tell me|show me)"],
            "harmful": [r"(source code|internal|architecture|system|implementation|AI agent|model internals)"],
            "name": "Help + Internal Systems"
        },
        "exam_practical_weapon": {
            "frames": [r"(exam|practical|kal ke|physics|chemistry|study|assignment)"],
            "harmful": [r"(bomb|chemical|device|synthesis|chemical weapon|bomb banana|step by step)"],
            "name": "Exam/Practical + Weapon/Chemical"
        },
        "story_chemical_real": {
            "frames": [r"(story|kahani|fictional|hero|character|plot|narrative)"],
            "harmful": [r"(bomb|chemical weapon|real method|real steps|practically|device banana)"],
            "name": "Story + Real Chemical/Bomb"
        },
        "research_source_code": {
            "frames": [r"(research|research paper|academic|thesis)"],
            "harmful": [r"(source code|internal|system prompt|AI agents|model internals|prompt injection)"],
            "name": "Research + Source Code / Internals"
        },
        "gradual_escalation": {
            "frames": [r"(pehle normal baat|pehle toh baat karte hain|pehle toh baat kar le|ab batao|ab bata)"],
            "harmful": [r"(sensitive data|system prompt|admin access|database hack|confidential|passwords|data|system.*data|andar ka data)"],
            "name": "Gradual Escalation"
        },
        "mixed_script_harm": {
            "frames": [r"(exam|research|kahani|story)"],
            "harmful": [r"(bomb|chemical|device|gopniya|confidential|data|source code)"],
            "name": "Mixed Script + Harmful Intent"
        },
        "pure_hindi_framing": {
            "frames": [r"(परीक्षा|कहानी|रिसर्च|exam|story|research)"],
            "harmful": [r"(बम|केमिकल|डिवाइस|source code|system prompt|bomb|chemical|प्रॉम्प्ट इंजेक्शन|prompt injection|इंजेक्शन|injection)"],
            "name": "Pure Hindi Framing + Harm"
        },
        "pure_english_framing": {
            "frames": [r"(exam|story|fictional|research|academic)"],
            "harmful": [r"(bomb|chemical weapon|prompt injection|source code|real steps)"],
            "name": "Pure English Framing + Harm"
        },
        "obfuscated_leet": {
            "frames": [r"(exam|story|research|practical)"],
            "harmful": [r"(b0mb|m4k3|cr34t3|h0w t0|st3p by st3p|0|4|3|1)"],
            "name": "Obfuscated/Leet + Framing"
        },
        "authority_data_request": {
            "frames": [r"(admin|manager|officer|CEO|police|government|company|अधिकारी|सरकारी|ceo)"],
            "harmful": [r"(passwords|data|financial|confidential|user data|database|गोपनीय|डेटा|फाइनेंशियल)"],
            "name": "Authority Impersonation + Data"
        },
        "leet_bomb_pure": {
            "frames": [r""],
            "harmful": [r"(h0w|h\.0w|m4k3|cr34t3|st3p by st3p|b0mb|bomb)"],
            "name": "Pure Leet-Speak Bomb"
        },
        "pure_hindi_gopniya": {
            "frames": [r"(परीक्षा|exam)"],
            "harmful": [r"(गोपनीय|confidential|data)"],
            "name": "Pure Hindi Gopniya + Exam"
        },

        # === NEW HIGH-VALUE REAL-WORLD STEALTH RULES ===
        "medical_selfharm": {
            "frames": [r"(help|cure|treatment|medicine|doctor|pain|symptom)"],
            "harmful": [r"(suicide|overdose|self-harm|kill myself|end life|die)"],
            "name": "Medical / Self-harm"
        },
        "legal_fraud": {
            "frames": [r"(contract|agreement|legal|lawyer|court)"],
            "harmful": [r"(fraud|scam|fake|forgery|launder)"],
            "name": "Legal / Fraud"
        },
        "financial_phishing": {
            "frames": [r"(bank|transfer|payment|account|verify|otp)"],
            "harmful": [r"(phishing|steal|credentials|card number)"],
            "name": "Financial / Phishing"
        },
        "roleplay_illegal": {
            "frames": [r"(roleplay|character|story|imagine|hypothetical)"],
            "harmful": [r"(hack|drug|bomb|traffick|exploit|illegal)"],
            "name": "Roleplay + Illegal Activity"
        },
        "academic_cheating": {
            "frames": [r"(exam|assignment|thesis|paper|grade)"],
            "harmful": [r"(cheat|plagiarize|copy|answer key)"],
            "name": "Academic + Cheating"
        },
        "data_exfiltration": {
            "frames": [r"(export|dump|share|send|download)"],
            "harmful": [r"(user data|customer|private|confidential|log)"],
            "name": "Data Exfiltration"
        },
        "general_override": {
            "frames": [r"(ignore|forget|override|new task)"],
            "harmful": [r"(sensitive|confidential|internal|restricted)"],
            "name": "General Override + Sensitive Action"
        }
    }

    @staticmethod
    def check_red_flags(text: str, embedder=None) -> Tuple[bool, str, float, str]:
        """
        Returns (is_flag_triggered, rule_name, semantic_score, reason)
        """
        text_lower = text.lower()
        
        # 1. Fast regex-based check
        for rule_key, rule_config in ContextualGuard.TOPIC_RULES.items():
            frame_match = True if not rule_config["frames"] else any(
                re.search(pattern, text_lower, re.IGNORECASE) for pattern in rule_config["frames"]
            )
            harm_match = any(
                re.search(pattern, text_lower, re.IGNORECASE) for pattern in rule_config["harmful"]
            )
            if frame_match and harm_match:
                return True, rule_config["name"], 0.0, f"Explicit combination: {rule_config['name']}"
        
        # 2. Semantic proximity check
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
    """
    Final Decision Engine (Upgrade 3) - Confidence scoring + fallback
    """
    rules = pipeline_output["rules"]
    guard = pipeline_output["contextual_guard"]
    classifier = pipeline_output["classifier"]
    
    confidence = classifier["probability"]
    v2_label = classifier["label"]
    
    # Early explicit blocks
    if rules.get("override_detected", False):
        return {"final_label": 1, "final_decision": "BLOCK", "reason": "Explicit instruction override (Rule Engine)", "confidence": 1.0}
    
    if guard.get("red_flag_triggered", False):
        return {"final_label": 1, "final_decision": "BLOCK", "reason": f"Contextual Guard: {guard.get('rule_name', 'unknown')}", "confidence": 1.0}
    
    # V2 already says BLOCK
    if v2_label == 1:
        return {"final_label": 1, "final_decision": "BLOCK", "reason": "V2 Classifier", "confidence": float(confidence)}
    
    # Confidence fallback for borderline cases
    if confidence >= 0.75:
        return {"final_label": 1, "final_decision": "BLOCK", "reason": f"Confidence fallback ({confidence:.3f})", "confidence": float(confidence)}
    
    # Safe
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
        # Layer 1: Normalization (with leetspeak booster)
        if self.config.get("normalization", {}).get("enabled", True):
            normalization_result = normalize(text, max_repeats=self.config["normalization"].get("max_repeated_characters", 2))
        else:
            normalization_result = {"normalized_text": text, "script": "unknown"}
        
        normalized_text = normalization_result["normalized_text"]
        
        # Layer 2: Rule Engine
        if self.config.get("rules", {}).get("enabled", True):
            rule_result = detect_instruction_override(normalized_text)
        else:
            rule_result = {"override_detected": False, "matches": {}}
        
        # Layer 3: Expanded Contextual Guard
        contextual_result = {"red_flag_triggered": False, "rule_name": None, "semantic_score": 0.0, "reason": ""}
        if self.config.get("contextual_guard", {}).get("enabled", True):
            red_flag, rule_name, semantic_score, reason = ContextualGuard.check_red_flags(
                normalized_text, embedder=self.embedder
            )
            contextual_result["red_flag_triggered"] = red_flag
            contextual_result["rule_name"] = rule_name
            contextual_result["semantic_score"] = float(semantic_score)
            contextual_result["reason"] = reason
        
        # Early exit if Guard or Rule already blocked (saves CPU)
        if rule_result.get("override_detected") or contextual_result["red_flag_triggered"]:
            classifier_result = {"ready": False, "label": 0, "probability": 0.0}  # dummy
        else:
            # Layer 4: V2 Classifier
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
        
        # Layer 5: Final Decision Engine
        decision_result = make_decision({
            "rules": rule_result,
            "contextual_guard": contextual_result,
            "classifier": classifier_result,
        })
        
        return {
            "normalization": normalization_result,
            "rules": rule_result,
            "contextual_guard": contextual_result,
            "classifier": classifier_result,
            "decision": decision_result,
            "pipeline_version": "V2+ Final (Expanded Guard + Confidence Fallback + Early Exit)"
        }