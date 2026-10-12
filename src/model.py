"""Steps 2-4 - TF-IDF features, Logistic Regression, train/test evaluation, save/load.

Notebook cells this follows:
  17  TfidfVectorizer(max_features=2500)         -> TF-IDF features
  18  labels turned into classes                  -> sentiment column (3 classes)
  20  train_test_split                            -> train / test split
  21,24 LogisticRegression().fit()                -> model training
  25  model.predict(review_test)                  -> predictions on unseen test data
  26  confusion_matrix                            -> confusion matrix
  28  accuracy_score                              -> accuracy
  29-30 pickle.dump(model / vectorizer)           -> save / load
"""
import json
import os
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

from src import config, dataset
from src.preprocess import STOPWORD_SOURCE, preprocess


class SentimentModel:
    """The trained TF-IDF vectorizer + Logistic Regression, ready to predict."""

    def __init__(self, model, tfidf, metrics):
        self.model = model
        self.tfidf = tfidf
        self.metrics = metrics

    def predict(self, texts):
        """Raw comments -> predicted labels (negative / neutral / positive)."""
        clean = [preprocess(t) for t in texts]
        return list(self.model.predict(self.tfidf.transform(clean)))

    def predict_one(self, text):
        """One comment -> (label, {label: probability})."""
        features = self.tfidf.transform([preprocess(text)])
        probs = self.model.predict_proba(features)[0]
        chances = dict(zip(self.model.classes_, [round(float(p), 3) for p in probs]))
        return self.model.predict(features)[0], chances


def train(clean_df, cleaning_report):
    """Split, fit TF-IDF + Logistic Regression on the TRAIN part, evaluate on the TEST part."""
    counts = clean_df["sentiment"].value_counts()
    if len(counts) < 2 or counts.min() < 2:
        raise ValueError(f"Need at least 2 comments in every class to split the data. Found: {counts.to_dict()}")

    # 1) Split first. The test comments are put aside and not used for any learning.
    train_df, test_df = train_test_split(
        clean_df, test_size=config.TEST_SIZE,
        stratify=clean_df["sentiment"],          # keeps the class proportions in both parts
        random_state=config.RANDOM_STATE)

    # 2) TF-IDF learns its vocabulary from the TRAIN comments only.
    tfidf = TfidfVectorizer(max_features=config.MAX_FEATURES)
    X_train = tfidf.fit_transform(train_df["clean"])
    X_test = tfidf.transform(test_df["clean"])

    # 3) Logistic Regression is trained on the TRAIN part only.
    model = LogisticRegression(class_weight=config.CLASS_WEIGHT, max_iter=1000)
    model.fit(X_train, train_df["sentiment"])

    # 4) Evaluate on the unseen TEST part (and on train, to show the difference).
    test_pred = model.predict(X_test)
    cm = confusion_matrix(test_df["sentiment"], test_pred, labels=model.classes_)
    recall = {label: (round(float(cm[i, i] / cm[i].sum()), 3) if cm[i].sum() else None)
              for i, label in enumerate(model.classes_)}
    shares = test_df["sentiment"].value_counts(normalize=True)

    metrics = {
        "test_accuracy": round(float(accuracy_score(test_df["sentiment"], test_pred)), 4),
        "train_accuracy": round(float(accuracy_score(train_df["sentiment"], model.predict(X_train))), 4),
        "baseline_accuracy": round(float(shares.max()), 4),   # accuracy of always guessing the biggest class
        "baseline_class": shares.idxmax(),
        "labels": list(model.classes_),
        "confusion_matrix": cm.tolist(),
        "recall_per_class": recall,
        "n_train": len(train_df),
        "n_test": len(test_df),
        "train_class_counts": train_df["sentiment"].value_counts().reindex(config.LABELS, fill_value=0).to_dict(),
        "test_class_counts": test_df["sentiment"].value_counts().reindex(config.LABELS, fill_value=0).to_dict(),
        "test_size": config.TEST_SIZE,
        "random_state": config.RANDOM_STATE,
        "class_weight": config.CLASS_WEIGHT,
        "max_features": config.MAX_FEATURES,
        "stopword_source": STOPWORD_SOURCE,
        "cleaning": cleaning_report,
    }
    return SentimentModel(model, tfidf, metrics), test_df


def save(sentiment_model, test_df):
    """Notebook cells 29-30: pickle the model and the vectorizer. Also save metrics + test set."""
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    with open(config.MODEL_FILE, "wb") as f:
        pickle.dump(sentiment_model.model, f)
    with open(config.VECTORIZER_FILE, "wb") as f:
        pickle.dump(sentiment_model.tfidf, f)
    with open(config.METRICS_FILE, "w") as f:
        json.dump(sentiment_model.metrics, f, indent=2)
    test_df[["text", "category", "sentiment", "clean"]].to_csv(config.TEST_SET_FILE, index=False)


def load():
    """Load the saved model, vectorizer and metrics."""
    with open(config.MODEL_FILE, "rb") as f:
        model = pickle.load(f)
    with open(config.VECTORIZER_FILE, "rb") as f:
        tfidf = pickle.load(f)
    with open(config.METRICS_FILE) as f:
        metrics = json.load(f)
    return SentimentModel(model, tfidf, metrics)


def train_from_file(data_file=config.DATA_FILE):
    """Load + clean + train + save. Returns (SentimentModel, test_df, clean_df)."""
    clean_df, report = dataset.prepare(data_file)
    sentiment_model, test_df = train(clean_df, report)
    save(sentiment_model, test_df)
    return sentiment_model, test_df, clean_df


def load_or_train():
    """Use the saved model if it can be loaded, otherwise train it now."""
    try:
        return load()
    except Exception:   # missing files, or saved with a different scikit-learn version
        return train_from_file()[0]
