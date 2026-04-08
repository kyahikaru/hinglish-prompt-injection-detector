import pandas as pd
import pickle
import yaml
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()
    classifier_model_path = config["classifier"]["model_path"]
    model_dir = os.path.dirname(classifier_model_path) or "models"
    model_stem = os.path.splitext(os.path.basename(classifier_model_path))[0]
    vectorizer_path = os.path.join(model_dir, f"{model_stem}_vectorizer.pkl")
    model_path = os.path.join(model_dir, f"{model_stem}.pkl")

    # Load dataset
    data = pd.read_csv(config["data"]["dataset_path"])

    X = data["text"]
    y = data["label"]

    # Vectorize text
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.9
    )
    X_vec = vectorizer.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_vec,
        y,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"]
    )

    # Train classifier
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Save artifacts
    with open(vectorizer_path, "wb") as f:
        pickle.dump(vectorizer, f)

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print("Model and vectorizer saved.")


if __name__ == "__main__":
    main()
