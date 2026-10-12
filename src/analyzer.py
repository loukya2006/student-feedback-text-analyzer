"""Connects the modules: comments -> preprocessing -> MODEL PREDICTIONS -> counts, words, categories.

Every sentiment shown in the dashboard / report comes from SentimentModel.predict().
"""
import pandas as pd

from src.config import LABELS
from src.keywords import top_words
from src.preprocess import preprocess


def run_analysis(df, sentiment_model):
    """df needs a 'text' column. Returns a dict of results for the dashboard / report."""
    df = df.copy()
    df["clean"] = df["text"].apply(preprocess)
    df["predicted"] = sentiment_model.predict(df["text"])          # <- the trained model

    counts = df["predicted"].value_counts().reindex(LABELS, fill_value=0)
    result = {
        "df": df,
        "counts": counts,
        "percent": (counts / len(df) * 100).round(1),
        "top_words": top_words(df["clean"], 12),
        "words_by_label": {label: top_words(df.loc[df["predicted"] == label, "clean"], 8) for label in LABELS},
        "by_category": None,
        "actual_counts": None,
    }
    if "category" in df.columns:
        table = pd.crosstab(df["category"], df["predicted"]).reindex(columns=LABELS, fill_value=0)
        result["by_category"] = table
    # If the file also contains true labels (e.g. the held-out test set), keep them for comparison.
    if "sentiment" in df.columns and df["sentiment"].notna().all() and set(df["sentiment"]) <= set(LABELS):
        result["actual_counts"] = df["sentiment"].value_counts().reindex(LABELS, fill_value=0)
    return result
