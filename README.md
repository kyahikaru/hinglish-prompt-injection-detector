# Hinglish Prompt Injection Detector

## Overview

This repository contains an implementation of a prompt injection detection system designed for Hinglish chatbots. Hinglish is a commonly used mix of Hindi and English that introduces irregular spelling, transliteration, and language switching, making traditional English-based security filters unreliable.

The system follows a layered detection approach inspired by the accompanying research work. It combines Hinglish-specific text normalization, rule-based detection of instruction manipulation, and a machine learning classifier to identify both direct and subtle prompt injection attempts in real time.

## Problem Statement

Prompt injection attacks allow users to embed hidden instructions inside normal-looking inputs, causing chatbots to ignore system constraints or safety rules. Detecting such attacks becomes significantly harder in Hinglish conversations due to inconsistent spelling, mixed scripts, and informal sentence structure.

Most existing prompt injection defenses are designed for clean English text and fail to generalize to code-mixed language settings. This creates a gap in safety for Hinglish-based chatbot applications used in education, customer support, and public services.

## Threat Model

The system focuses on detecting direct prompt injection attempts in single-turn user inputs. These attacks involve users embedding hidden or explicit instructions intended to override or manipulate the chatbot’s intended behavior.

Out of scope are multi-turn attacks, retrieval-based prompt injection, and tool-augmented exploitation. The goal is to provide a lightweight and practical defense for real-time Hinglish chatbot interactions.

## System Architecture

The detection pipeline is composed of three main stages:

1. **Hinglish Text Normalization**  
   Preprocesses input to reduce noise caused by transliteration, spelling variations, and mixed scripts.

2. **Rule-Based Instruction Detection**  
   Identifies explicit patterns that indicate attempts to override system instructions or control chatbot behavior.

3. **Machine Learning Classifier**  
   Analyzes the semantic intent of the input to detect more subtle injection attempts that are not obvious at the surface level.

The final decision is made by combining the outputs of these layers to determine whether an input is safe or unsafe.

## Evaluation Criteria

The system is evaluated using the following metrics:

- Precision
- Recall
- F1 Score
- False Positive Rate
- Latency

These metrics are chosen to balance security effectiveness with usability and real-time performance.

## Limitations

This implementation does not handle multi-turn conversations or indirect prompt injection through external knowledge sources. Advanced obfuscation techniques and adaptive adversarial attacks are also outside the current scope.

The system is intended as a practical baseline for Hinglish prompt injection detection rather than a complete security solution.

## How to Run

1. Clone the repository:

2. Install dependencies:

3. Run the detector:

4. Enter a sample Hinglish input when prompted.

