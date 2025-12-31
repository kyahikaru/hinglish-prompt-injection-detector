# Detection pipeline orchestration
from typing import Dict
import yaml

from preprocessing.normalization import normalize
from rules.instruction_rules import detect_instruction_override
from models.classifier import SemanticClassifier


class DetectionPipeline:
    """
    Orchestrates the prompt injection detection pipeline.
    """

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the pipeline, load configuration and classifier.
        """
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.classifier = SemanticClassifier()

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
        rule_result = detect_instruction_override(normalized_text)

        # Step 3: Semantic classification
        classifier_result = self.classifier.predict_proba(normalized_text)

        return {
            "normalization": normalization_result,
            "rules": rule_result,
            "classifier": classifier_result
        }
