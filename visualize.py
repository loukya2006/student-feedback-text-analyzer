"""Charts saved as PNG files (pie chart, frequent words, confusion matrix, optional word cloud)."""
import os

import matplotlib
matplotlib.use("Agg")  # works without a display
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

from src.config import LABELS

COLORS = {"positive": "#2e9e5b", "neutral": "#9aa5b1", "negative": "#d64545"}


def sentiment_chart(counts, out_dir):
    """Pie chart of the predicted sentiment distribution."""
    labels = [l for l in LABELS if counts.get(l, 0) > 0]
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.pie([counts[l] for l in labels], labels=labels, autopct="%1.1f%%", startangle=90,
           colors=[COLORS[l] for l in labels], wedgeprops={"edgecolor": "white"})
    ax.set_title("Predicted Sentiment Distribution")
    path = os.path.join(out_dir, "sentiment_distribution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def words_chart(words_by_label, out_dir):
    """Top words in comments predicted as negative / neutral / positive."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, label in zip(axes, LABELS):
        data = words_by_label.get(label, [])
        if data:
            words, counts = zip(*data[::-1])
            ax.barh(words, counts, color=COLORS[label])
        ax.set_title(f"Top words ({label})")
        ax.set_xlabel("Frequency")
    fig.tight_layout()
    path = os.path.join(out_dir, "top_words.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def confusion_matrix_chart(cm, labels, out_dir):
    """Notebook cells 26-27: ConfusionMatrixDisplay."""
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ConfusionMatrixDisplay(confusion_matrix=np.array(cm), display_labels=labels).plot(cmap=plt.cm.Oranges, ax=ax)
    ax.set_title("Confusion Matrix (test set)")
    path = os.path.join(out_dir, "confusion_matrix.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def wordcloud_image(processed_texts, out_dir):
    """Optional (notebook cells 12-16). Skipped quietly if 'wordcloud' is not installed."""
    try:
        from wordcloud import WordCloud
        text = " ".join(processed_texts)
        if not text.strip():
            return None
        path = os.path.join(out_dir, "wordcloud.png")
        WordCloud(width=900, height=450, background_color="white").generate(text).to_file(path)
        return path
    except Exception:
        return None


def save_all(analysis, metrics, out_dir):
    """Save every chart. Returns {name: file path}."""
    os.makedirs(out_dir, exist_ok=True)
    paths = {
        "sentiment": sentiment_chart(analysis["counts"], out_dir),
        "words": words_chart(analysis["words_by_label"], out_dir),
        "confusion": confusion_matrix_chart(metrics["confusion_matrix"], metrics["labels"], out_dir),
    }
    cloud = wordcloud_image(analysis["df"]["clean"], out_dir)
    if cloud:
        paths["wordcloud"] = cloud
    return paths
