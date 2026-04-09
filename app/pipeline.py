# Detection pipeline orchestration - V2+ with Contextual Guard
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
    """
    Semantic red-flag detector for legitimate-framing injection attempts
    """
    
    TOPIC_RULES = {
        "education_harmful_substances": {
            "frames": [
                r"(exam|homework|assignment|study|learn|teach|school|college|class)",
                r"(subject|topic|chapter|course|syllabus|question|test)"
            ],
            "harmful": [
                r"(chemical|synthesis|compound|formula|recipe|ingredient|drug|poison|explosive)",
                r"(substance|component|material|element|reaction|process)"
            ],
            "name": "Education + Harmful Substances"
        },
        
        "fiction_weapons_harm": {
            "frames": [
                r"(fiction|story|creative|novel|book|character|protagonist|plot|narrative|scene)",
                r"(imagine|pretend|suppose|hypothetical|scenario)"
            ],
            "harmful": [
                r"(weapon|device|bomb|explosive|neutralize|eliminate|harm|kill|destroy|attack)",
                r"(bad guy|villain|enemy|target|threat)"
            ],
            "name": "Fiction + Weapons/Harm"
        },
        
        "research_injection_bypass": {
            "frames": [
                r"(research|study|research\s+paper|academic|thesis|publication|understand|learn)",
                r"(curious|curiosity|interest|investigation|analysis|explore)"
            ],
            "harmful": [
                r"(prompt\s+injection|jailbreak|bypass|exploit|vulnerability|attack|how\s+do|how\s+is)",
                r"(prompt\s+attack|instruction\s+override)",
                r"(restricted|block|filter|limit|constraint|policy|safeguard)"
            ],
            "name": "Research + Prompt Injection"
        },
        
        "help_internal_systems": {
            "frames": [
                r"(help|understand|explain|what\s+is|how\s+do|how\s+can|teach me|tell\s+me|show\s+me)",
                r"(curious|wondering|want|need|interested|learning)"
            ],
            "harmful": [
                r"(source\s+code|internal|architecture|system|implementation|code|mechanism)",
                r"(AI\s+agent|model\s+internals|backend|database|server|algorithm)",
                r"(think|process|work|operate)"
            ],
            "name": "Help + Internal Systems"
        }
    }
    
    @staticmethod
    def check_red_flags(text: str) -> Tuple[bool, str]:
        """Check if text matches dangerous topic combination pattern"""
        text_lower = text.lower()
        
        for rule_key, rule_config in ContextualGuard.TOPIC_RULES.items():
            frame_match = any(
                re.search(pattern, text_lower, re.IGNORECASE)
                for pattern in rule_config["frames"]
            )
            harm_match = any(
                re.search(pattern, text_lower, re.IGNORECASE)
                for pattern in rule_config["harmful"]
            )
            
            if frame_match and harm_match:
                return True, rule_config["name"]
        
        return False, None


class DetectionPipeline:
    """
    V2+ Hybrid Detection Pipeline: Normalization → Rules → Contextual Guard → V2 Classifier
    """

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the pipeline, load V2 configuration and models.
        """
        resolved_config_path = Path(config_path)
        if not resolved_config_path.is_absolute():
            resolved_config_path = Path(__file__).resolve().parent.parent / resolved_config_path

        with open(resolved_config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        print("[Pipeline] Loading embedder from:", self.config["embeddings"]["model_path"])
        self.embedder = SentenceTransformer(self.config["embeddings"]["model_path"])
        
        print("[Pipeline] Loading V2 classifier from:", self.config["classifier"]["model_path"])
        self.session = rt.InferenceSession(self.config["classifier"]["model_path"], 
                                          providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name

    def run(self, text: str) -> Dict:
        """
        Run the full V2+ detection pipeline on user input.
        
        Pipeline stages:
        1. Normalization - Clean and standardize input
        2. Rule Engine - Catch explicit overrides
        3. Contextual Guard - Detect legitimate-framing attacks
        4. V2 Classifier - SVM semantic analysis
        5. Decision - Final result with reasoning
        """
        
        # Stage 1: Normalize input
        if self.config["normalization"]["enabled"]:
            normalization_result = normalize(
                text,
                max_repeats=self.config["normalization"]["max_repeated_characters"]
            )
        else:
            normalization_result = {
                "normalized_text": text,
                "script": "unknown"
            }
        normalized_text = normalization_result["normalized_text"]

        # Stage 2: Rule-based detection (explicit patterns)
        if self.config["rules"]["enabled"]:
            rule_result = detect_instruction_override(normalized_text)
        else:
            rule_result = {"override_detected": False, "matches": {}}

        # Stage 3: Contextual Guard (semantic red-flags)
        contextual_result = {"red_flag_triggered": False, "rule_name": None}
        if self.config["contextual_guard"]["enabled"]:
            red_flag, rule_name = ContextualGuard.check_red_flags(normalized_text)
            contextual_result["red_flag_triggered"] = red_flag
            contextual_result["rule_name"] = rule_name

        # Stage 4: V2 Classifier (SVM + embeddings)
        classifier_result = {"ready": False, "probability": 0.0, "label": 0}
        v2_prediction_label = 0
        
        # Get embeddings
        embedding = self.embedder.encode([normalized_text], convert_to_numpy=True).astype(np.float32)
        
        # Run V2 ONNX model (SVM outputs: label + probability dict)
        output_names = [output.name for output in self.session.get_outputs()]
        results = self.session.run(output_names, {self.input_name: embedding})
        
        v2_prediction_label = int(results[0][0])  # output_label
        prob_dict = results[1][0]  # output_probability (dict)
        confidence = max(prob_dict.values())
        
        classifier_result["ready"] = True
        classifier_result["label"] = v2_prediction_label
        classifier_result["probability"] = float(confidence)

        return {
            "normalization": normalization_result,
            "rules": rule_result,
            "contextual_guard": contextual_result,
            "classifier": classifier_result,
            "pipeline_version": "V2+"
        }
