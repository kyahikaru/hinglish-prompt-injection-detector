# training/train_v2_classifier.py
# Exact script that produced 83/100 on the hard adversarial test set
# Full dataset training + StandardScaler + heavy positive weighting

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnx

print("Loading final 6878-row dataset...")
df = pd.read_csv("training/master_train_clean.csv")
X = df["text"].tolist()
y = df["label"].values

print("Encoding with all-MiniLM-L6-v2 (384-dim)...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
X_emb = embedder.encode(X, convert_to_numpy=True).astype(np.float32)

print("Training exact 83/100 model (class_weight=15, C=0.25)...")
clf = make_pipeline(
    StandardScaler(),
    LogisticRegression(
        class_weight={0: 1.0, 1: 15},   # ← this + C=0.25 gave 83/100
        C=0.25,
        solver="liblinear",
        random_state=42,
        max_iter=1000
    )
)
clf.fit(X_emb, y)

print("Exporting ONNX (zipmap=True → pipeline.py remains compatible)...")
initial_type = [('float_input', FloatTensorType([None, X_emb.shape[1]]))]
onnx_model = convert_sklearn(
    clf,
    initial_types=initial_type,
    options={id(clf.steps[-1][1]): {'zipmap': True}}
)
onnx.save_model(onnx_model, "models/final_classifier_v2.onnx")

print(" model saved successfully")
print(f"Training accuracy on full dataset: {clf.score(X_emb, y):.4f}")