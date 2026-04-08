from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()
    dataset_path = config["data"]["dataset_path"]
    test_size = config["data"]["test_size"]
    random_state = config["data"]["random_state"]
    embedder_name = config["embeddings"]["model_name"]
    onnx_model_path = config["classifier"]["model_path"]

    data = pd.read_csv(dataset_path)
    X = data["text"].astype(str)
    y = data["label"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    embedder = SentenceTransformer(embedder_name)
    X_train_emb = embedder.encode(X_train.tolist()).astype(np.float32)
    X_test_emb = embedder.encode(X_test.tolist()).astype(np.float32)

    model = LogisticRegression()
    model.fit(X_train_emb, y_train)

    y_pred = model.predict(X_test_emb)
    print(classification_report(y_test, y_pred))

    initial_types = [("float_input", FloatTensorType([None, X_train_emb.shape[1]]))]
    onnx_model = convert_sklearn(model, initial_types=initial_types, options={id(model): {"zipmap": False}})

    output_path = Path(onnx_model_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(onnx_model.SerializeToString())

    print(f"Saved ONNX classifier to: {output_path}")


if __name__ == "__main__":
    main()
