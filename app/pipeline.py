# Detection pipeline orchestration will go here
from typing import Dict

from preprocessing.normalization import normalize
from rules.instruction_rules import detect_instruction_override
from models.classifier import SemanticClassifier


class DetectionPipeline:
    """
    Orchestrates the prompt injection detection pipeline.
    """

    def __init__(self, classifier: SemanticClassifier):
        self.classifier = classifier

    def run(self, text: str) -> Dict:
        """
        Run the full detection pipeline on user input.
        """
        # Step 1: Normalize input
        normalization_result = normalize(text)
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
