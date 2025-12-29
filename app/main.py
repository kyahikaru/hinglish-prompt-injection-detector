# Entry point for running the detector
import yaml

from models.classifier import SemanticClassifier
from app.pipeline import DetectionPipeline
from app.decision import make_decision


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    # Load configuration
    config = load_config()

    # Initialize semantic classifier (model loading handled later)
    classifier = SemanticClassifier(
        model=None,
        vectorizer=None
    )

    # Initialize detection pipeline
    pipeline = DetectionPipeline(classifier)

    # Read input
    user_input = input("Enter user input: ")

    # Run pipeline
    pipeline_result = pipeline.run(user_input)

    # Make final decision
    decision = make_decision(
        rule_result=pipeline_result["rules"],
        classifier_result=pipeline_result["classifier"],
        probability_threshold=config["classifier"]["probability_threshold"]
    )

    # Output result
    output = {
        "pipeline": pipeline_result,
        "decision": decision
    }

    print(output)


if __name__ == "__main__":
    main()
