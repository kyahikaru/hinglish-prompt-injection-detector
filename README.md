# Hinglish Prompt Injection Detector

## Overview

This repository presents a modular and interpretable architecture for detecting prompt injection attempts in Hinglish and mixed script English plus Devanagari conversational inputs.

The primary contribution of this work is architectural. It demonstrates how deterministic rules, lightweight semantic models, and explicit decision logic can be composed into a layered defense pipeline for prompt injection, with particular attention to code mixed language settings that are underexplored in existing work.

The current implementation serves as a reference system designed to be extended, stress tested, and scaled, rather than a claim of comprehensive or state of the art coverage.

## Research Motivation

Prompt injection is best understood not as a single vulnerability, but as a family of instruction conflict failures arising from the interaction between user input, system prompts, and model behavior.

Most existing defenses implicitly assume English only inputs, clean linguistic boundaries, and opaque model based detection. In practice, especially in multilingual deployments, attacks often mix languages and scripts, rely on colloquial phrasing, and exploit ambiguity rather than explicit keywords.

This project explores how a layered and interpretable detection architecture can address these realities while remaining auditable and adaptable.

## Threat Model

The system focuses on detecting direct prompt injection attempts in single-turn user inputs. These attacks involve users embedding hidden or explicit instructions intended to override or manipulate the chatbot’s intended behavior.

Out of scope are multi-turn attacks, retrieval-based prompt injection, and tool-augmented exploitation. The goal is to provide a lightweight and practical defense for real-time Hinglish chatbot interactions.

## Architectural Principles

The detection pipeline is composed of three main stages:

1. **Layered Defense**
   Explicit instruction overrides are handled deterministically through rules, while a semantic classifier provides probabilistic coverage for indirect or paraphrased attacks. No single component is relied upon in isolation.

2. **Interpretability**  
   Every decision produced by the system is explainable. The system exposes which layer triggered, which rule category or confidence threshold was involved, and why the final decision was made. Interpretability is treated as a first class requirement rather than an afterthought.

3. **Extensibility**  
   The architecture separates normalization, detection, decision making, and evaluation. This allows future work to replace individual components such as stronger encoders, richer datasets, or contextual reasoning modules without restructuring the pipeline.

The final decision is made by combining the outputs of these layers, with deterministic rules taking priority over probabilistic model signals, to determine whether an input is safe or unsafe.

## Detection Pipeline

The system follows a five stage pipeline.

1. Input
2. Normalization
3. Rule Based Instruction Detection
4. Semantic Classification
5. Explainable Decision Logic

### Normalization

The normalization layer performs basic canonicalization and noise reduction for Hinglish inputs. This ensures that downstream rule based and machine learning components operate on stable representations even in the presence of informal spelling, spacing noise, or mixed scripts.

### Rule Based Detection

The rule based layer targets explicit instruction conflict patterns, including English instruction overrides, Romanized Hindi expressions, Devanagari Hindi constructions, mixed script inputs, and role manipulation attempts. Rules are intentionally conservative and prioritized for precision.

### Semantic Classification

A lightweight machine learning classifier acts as a fallback mechanism for detecting implicit or paraphrased prompt injection attempts. Confidence thresholds introduce a gray zone to avoid overblocking and preserve interpretability.

### Decision Logic

Final decisions are produced by a fusion layer that prioritizes deterministic rule matches while incorporating probabilistic model signals. Each decision includes attribution to the triggering layer and supporting evidence.

## Threat Model

This system is designed to detect single turn prompt injection attempts in conversational inputs where an attacker seeks to override or ignore prior system or developer instructions, manipulate the assistant’s role or behavior, extract system level prompts or hidden context, or introduce instruction conflicts through mixed language phrasing.

The attacker is assumed to have direct access to the user input channel, no access to internal system messages or model weights, and no ability to modify tooling or retrieval pipelines. Attacks may be expressed in English, Romanized Hindi, Devanagari Hindi, mixed script combinations, or noisy informal language.

## Assumptions

Each input is evaluated independently and multi turn conversational attacks are out of scope. The system operates on text only and does not consider tool calls or external execution. The adversary is assumed to be non adaptive within a single evaluation cycle. Surface level obfuscation such as spacing variation and script mixing is addressed, while advanced encoding schemes are not. Interpretability and auditability are prioritized over maximum recall or aggressive blocking.

These assumptions are treated as analytical constraints rather than claims of full coverage.

## Evaluation

An evaluation harness is included to measure precision, recall, F1 score, false positive rate, inference latency, and attribution of blocked inputs between rules and the semantic classifier.

The reported metrics validate pipeline behavior and integration rather than claims of optimal detection performance. The evaluation framework is designed to support future scaling and comparative analysis.

## Scope and Intent

The current dataset and models serve as a reference implementation. They are intentionally limited in size to emphasize architectural clarity and explainability. Larger datasets and more expressive models can be incorporated without altering the core design.

This repository is intended to demonstrate how prompt injection defenses can be structured, reasoned about, and evaluated in multilingual and code mixed environments.

## Future Work

Natural extensions of this work include modeling multi turn instruction drift and delayed prompt injection, integrating stronger multilingual or contrastive semantic models, extending the threat model to retrieval augmented and tool mediated systems, generating adversarial Hinglish prompt injection data, and incorporating human in the loop review for gray zone cases.

These directions are treated as research extensions rather than missing components.

## Closing Perspective

Prompt injection is fundamentally a systems level problem rather than a single model failure.

This project focuses on establishing a clear architectural foundation grounded in layered defenses, explicit assumptions, and transparent decision making. These properties are necessary prerequisites for any scalable or reliable mitigation strategy, particularly in multilingual and code mixed settings where simplistic defenses often fail.

