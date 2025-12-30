import pandas as pd
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def main():
    # Load dataset
    data = pd.read_csv("training/dataset.csv")

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
        X_vec, y, test_size=0.2, random_state=42
    )

    # Train classifier
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Save artifacts
    with open("models/vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    with open("models/model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("Model and vectorizer saved.")


if __name__ == "__main__":
    main()
