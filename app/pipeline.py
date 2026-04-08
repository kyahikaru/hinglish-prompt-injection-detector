# Detection pipeline orchestration
from typing import Dict
from pathlib import Path
import yaml
import numpy as np
import onnxruntime as rt
from sentence_transformers import SentenceTransformer

from preprocessing.normalization import normalize
from rules.instruction_rules import detect_instruction_override


class DetectionPipeline:
    """
    Orchestrates the prompt injection detection pipeline.
    """

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the pipeline, load configuration and classifier.
        """
        resolved_config_path = Path(config_path)
        if not resolved_config_path.is_absolute():
            resolved_config_path = Path(__file__).resolve().parent.parent / resolved_config_path

        with open(resolved_config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.embedder = SentenceTransformer(self.config["embeddings"]["model_name"])
        self.session = rt.InferenceSession(self.config["classifier"]["model_path"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[1].name

    def run(self, text: str) -> Dict:
        """
        Run the full detection pipeline on user input.
        """
        # Step 1: Normalize input (config-controlled)
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

        # Step 2: Rule-based detection
        if self.config["rules"]["enabled"]:
            rule_result = detect_instruction_override(normalized_text)
        else:
            rule_result = {"override_detected": False, "matches": {}}

        # Step 3: Semantic classification (ONNX + MiniLM)
        classifier_result = {"ready": True, "probability": 0.0}
        if not rule_result["override_detected"]:
            embedding = self.embedder.encode([normalized_text]).astype(np.float32)
            probabilities = self.session.run([self.output_name], {self.input_name: embedding})[0]
            classifier_result["probability"] = float(probabilities[0][1])

        return {
            "normalization": normalization_result,
            "rules": rule_result,
            "classifier": classifier_result
        }
