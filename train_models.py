import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# ========= READ TEXT RESUME FILE =========

def read_text_resume(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except:
        return ""

# ========= MAIN TRAINING FUNCTION =========

def train_model():
    labels_df = pd.read_csv("data/labels.csv")

    texts = []
    targets = []

    for idx, row in labels_df.iterrows():
        file_path = os.path.join("data/resumes", row["file_name"])

        text = read_text_resume(file_path)

        if text.strip() == "":
            print(f"⚠ Empty or unreadable: {row['file_name']}")
            continue

        texts.append(text)
        targets.append(row["label"])

    # Convert to DataFrame
    df = pd.DataFrame({"text": texts, "label": targets})

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42
    )

    # Vectorizer
    vectorizer = TfidfVectorizer(stop_words="english")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Model
    model = LogisticRegression(max_iter=200)
    model.fit(X_train_vec, y_train)

    # Accuracy
    preds = model.predict(X_test_vec)
    acc = accuracy_score(y_test, preds)

    print(f"\n🎉 Training complete!")
    print(f"📊 Accuracy: {acc * 100:.2f}%")

    # Save model + vectorizer
    joblib.dump(model, "resume_model.pkl")
    joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

    print("\n✔ resume_model.pkl saved")
    print("✔ tfidf_vectorizer.pkl saved")


if __name__ == "__main__":
    train_model()
