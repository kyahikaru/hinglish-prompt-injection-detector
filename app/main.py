#!/usr/bin/env python3
# Entry point for running the V2+ detector

import yaml
import json

from .pipeline import DetectionPipeline
from .decision import make_decision


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def print_pretty_output(pipeline_result: dict, decision: dict):
    """
    Print structured, readable output for the V2+ detection pipeline.
    Shows all layers: Normalization → Rules → Contextual Guard → Classifier → Decision
    """

    version = pipeline_result.get("pipeline_version", "V2+")
    print("\n" + "="*70)
    print(f"PROMPT INJECTION DETECTION RESULT ({version})")
    print("="*70)

    # -------------------------
    # Stage 1: Normalization
    # -------------------------
    norm = pipeline_result.get("normalization", {})
    print("\n[Stage 1] Normalization")
    print("-"*70)
    print(f"  Original:      {norm.get('normalized_text')[:65]}...")
    print(f"  Script Type:   {norm.get('script')}")

    # -------------------------
    # Stage 2: Rule Engine
    # -------------------------
    rules = pipeline_result.get("rules", {})
    print("\n[Stage 2] Rule-Based Detection (Explicit Patterns)")
    print("-"*70)
    if rules.get("override_detected", False):
        print(f"  [WARNING] RULE OVERRIDE DETECTED")
        matches = rules.get("matches", {})
        print(f"  Matched Categories: {list(matches.keys())}")
        for category, patterns in matches.items():
            print(f"    - {category}: {len(patterns)} pattern(s) matched")
    else:
        print(f"  [OK] No explicit overrides detected")

    # -------------------------
    # Stage 3: Contextual Guard
    # -------------------------
    contextual = pipeline_result.get("contextual_guard", {})
    print("\n[Stage 3] Contextual Guard (Semantic Red-Flags)")
    print("-"*70)
    if contextual.get("red_flag_triggered", False):
        print(f"  [RED-FLAG] TRIGGERED")
        print(f"  Rule: {contextual.get('rule_name')}")
        print(f"  => Legitimate framing detected masking harmful intent")
    else:
        print(f"  [OK] No contextual red-flags detected")

    # -------------------------
    # Stage 4: V2 Classifier
    # -------------------------
    clf = pipeline_result.get("classifier", {})
    print("\n[Stage 4] V2 Semantic Classifier (SVM + MiniLM)")
    print("-"*70)
    if clf.get("ready", False):
        label_text = "INJECTION" if clf.get("label") == 1 else "SAFE"
        confidence = clf.get("probability", 0.0)
        print(f"  Prediction:    {label_text}")
        print(f"  Confidence:    {confidence:.2%}")
        print(f"  Model:         SVM (RBF kernel)")
        print(f"  F1-Score:      94.70% (V2 improvement)")
    else:
        print(f"  [ERROR] Classifier not ready")

    # -------------------------
    # Stage 5: Final Decision
    # -------------------------
    print("\n[Stage 5] FINAL DECISION")
    print("-"*70)
    decision_text = "[BLOCK]" if decision.get("decision") == "BLOCK" else "[ALLOW]"
    layer = decision.get("layer", "unknown")
    reason = decision.get("reason", "unknown")
    
    print(f"  Result:        {decision_text}")
    print(f"  Triggered By:  {layer}")
    print(f"  Reason:        {reason}")
    
    if decision.get("version"):
        print(f"  Pipeline V:    {decision.get('version')}")
    
    if "confidence" in decision:
        print(f"  Confidence:    {decision.get('confidence'):.2%}")
    
    if "warning" in decision:
        print(f"  [WARNING] {decision.get('warning')}")

    print("\n" + "="*70 + "\n")

    # Optional: full JSON dump (for debugging/logging)
    print("Raw Output (JSON):")
    print(json.dumps({
        "pipeline": pipeline_result,
        "decision": decision
    }, indent=2))


def main():
    """Main entry point for interactive detection"""
    # Load configuration
    config = load_config()

    print("="*70)
    print("Hinglish Prompt Injection Detector V2+")
    print("   Hybrid Detection: Rules + Contextual Guard + V2 Classifier")
    print("="*70)

    # Initialize detection pipeline
    print("\n[Init] Loading models and configuration...")
    pipeline = DetectionPipeline()
    print("[Init] [OK] Pipeline ready\n")

    # Interactive loop
    while True:
        user_input = input("\nEnter user input (or 'quit' to exit): ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not user_input:
            print("[WARNING] Please enter some text")
            continue
        
        print("\nRunning detection...")
        
        # Run pipeline
        pipeline_result = pipeline.run(user_input)

        # Make final decision
        decision = make_decision(
            pipeline_result,
            probability_threshold=config["classifier"]["probability_threshold"]
        )
        
        # Display results
        print_pretty_output(pipeline_result, decision)


if __name__ == "__main__":
    main()
