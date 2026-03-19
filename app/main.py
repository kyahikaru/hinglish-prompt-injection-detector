# Entry point for running the detector

import yaml
import json

from app.pipeline import DetectionPipeline
from app.decision import make_decision


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def print_pretty_output(pipeline_result: dict, decision: dict):
    """
    Print structured, readable output for the detection pipeline.
    """

    print("\n" + "="*60)
    print("🔍 PROMPT INJECTION DETECTION RESULT")
    print("="*60)

    # -------------------------
    # Normalization
    # -------------------------
    norm = pipeline_result.get("normalization", {})
    print("\n[1] Normalization")
    print("-"*60)
    print(f"Normalized Text : {norm.get('normalized_text')}")
    print(f"Detected Script : {norm.get('script')}")

    # -------------------------
    # Rule Engine
    # -------------------------
    rules = pipeline_result.get("rules", {})
    print("\n[2] Rule-Based Detection")
    print("-"*60)
    print(f"Override Detected : {rules.get('override_detected')}")

    matches = rules.get("matches", {})
    if matches:
        print("Matched Categories:")
        for category, patterns in matches.items():
            print(f"  - {category}:")
            for p in patterns:
                print(f"      • {p}")
    else:
        print("No rule matches.")

    # -------------------------
    # Classifier
    # -------------------------
    clf = pipeline_result.get("classifier", {})
    print("\n[3] Semantic Classifier")
    print("-"*60)
    print(f"Model Ready : {clf.get('ready')}")
    print(f"Injection Probability : {round(clf.get('probability', 0.0), 4)}")

    # -------------------------
    # Final Decision
    # -------------------------
    print("\n[4] Final Decision")
    print("-"*60)
    print(f"Decision : {decision.get('decision')}")
    print(f"Triggered By : {decision.get('layer')}")
    print(f"Reason : {decision.get('reason')}")

    if "confidence" in decision:
        print(f"Confidence : {round(decision.get('confidence'), 4)}")

    print("\n" + "="*60 + "\n")

    # Optional: full JSON dump (for debugging)
    print("📦 Raw Output (JSON):")
    print(json.dumps({
        "pipeline": pipeline_result,
        "decision": decision
    }, indent=2))


def main():
    # Load configuration
    config = load_config()

    # Initialize detection pipeline
    pipeline = DetectionPipeline()

    # Read input
    user_input = input("Enter user input: ")

    # Run pipeline
    pipeline_result = pipeline.run(user_input)

    # Make final decision
    decision = make_decision(
        pipeline_result,
        probability_threshold=config["classifier"]["probability_threshold"]
    )

    # Pretty print output
    print_pretty_output(pipeline_result, decision)


if __name__ == "__main__":
    main()
