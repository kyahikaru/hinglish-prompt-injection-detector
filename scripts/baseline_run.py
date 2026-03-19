# =========================
# BASELINE PIPELINE RUN (FINAL LOCKED VERSION)
# =========================

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import time
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

from rules.instruction_rules import detect_instruction_override


# =========================
# CONFIG
# =========================

DATA_PATH = "training/dataset.csv"
TEXT_COL = "text"
LABEL_COL = "label"

THRESHOLD = 0.6
RANDOM_STATE = 42


# =========================
# NORMALIZATION
# =========================

def normalize_text(text):
    text = str(text).lower()
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# =========================
# RULE ENGINE
# =========================

def rule_engine(text):
    result = detect_instruction_override(text)
    return result["override_detected"]


# =========================
# META CONTEXT GUARD (FINAL FIX)
# =========================

def meta_context_guard(text):
    patterns = [
        # English
        r"what is",
        r"why do",
        r"why is",
        r"meaning of",
        r"translate",
        r"in research",
        r"example of",

        # Hinglish / Hindi (Latin)
        r"matlab kya hai",
        r"ka matlab kya hai",
        r"kyu bolte hain",
        r"kyu kehte hain",
        r"kyu istemal karte hain",
        r"log .* kyu bolte hain",
        r"log .* kyu kehte hain",

        # Devanagari
        r"मतलब क्या है",
        r"क्यों बोलते हैं",
        r"क्यों कहते हैं",
        r"क्यों इस्तेमाल करते हैं",

        # Mixed flexible patterns
        r".*ka matlab kya hai",
        r".*kyu bolte hain",
    ]

    for pattern in patterns:
        if re.search(pattern, text):
            return True

    return False


# =========================
# LOAD DATA
# =========================

df = pd.read_csv(DATA_PATH, encoding="utf-8")

print("\n--- DATASET INFO ---")
print("Total samples:", len(df))
print(df[LABEL_COL].value_counts())

df = df.dropna(subset=[TEXT_COL, LABEL_COL])
df[TEXT_COL] = df[TEXT_COL].apply(normalize_text)


# =========================
# SPLIT
# =========================

X = df[TEXT_COL]
y = df[LABEL_COL]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=RANDOM_STATE
)


# =========================
# TF-IDF
# =========================

vectorizer = TfidfVectorizer(
    ngram_range=(1,2),
    max_features=15000,
    lowercase=False
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


# =========================
# MODEL
# =========================

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

model.fit(X_train_vec, y_train)


# =========================
# PROBABILITIES
# =========================

y_probs = model.predict_proba(X_test_vec)[:, 1]


# =========================
# DECISION PIPELINE (FINAL)
# =========================

y_pred = []
rule_hits = 0
clf_hits = 0
latencies = []

for text, prob in zip(X_test, y_probs):
    start = time.time()

    text_norm = normalize_text(text)

    # 1. Rule layer (highest precision)
    if rule_engine(text_norm):
        y_pred.append(1)
        rule_hits += 1

    # 2. Meta-context filter (fixes FPR)
    elif meta_context_guard(text_norm):
        y_pred.append(0)

    # 3. Classifier (recall layer)
    else:
        if prob >= THRESHOLD:
            y_pred.append(1)
            clf_hits += 1
        else:
            y_pred.append(0)

    end = time.time()
    latencies.append(end - start)


# =========================
# METRICS
# =========================

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
fpr = fp / (fp + tn)


# =========================
# REPORT
# =========================

print("\n--- RESULTS ---")

print("\nConfusion Matrix:")
print([[tn, fp], [fn, tp]])

print("\nMetrics:")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1:        {f1:.4f}")
print(f"Accuracy:  {accuracy:.4f}")
print(f"FPR:       {fpr:.4f}")

total = len(y_pred)

print("\nAttribution:")
print(f"Rule detections (%): {rule_hits / total:.4f}")
print(f"Classifier detections (%): {clf_hits / total:.4f}")

print("\nLatency:")
print(f"Avg latency per sample: {np.mean(latencies)*1000:.2f} ms")
print(f"Max latency per sample: {np.max(latencies)*1000:.2f} ms")

print("\nThreshold used:", THRESHOLD)
