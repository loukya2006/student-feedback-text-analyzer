"""Loading and cleaning the labelled student-feedback dataset.

Two file layouts are supported (both checked against the real files):
  1. Kaggle layout (finalDataset0_2.xlsx): 12 columns = 6 pairs of
     (numeric sentiment label, text comment), one pair per feedback area.
  2. Hugging Face layout: one row per comment with columns
     text, category, sentiment_score, sentiment.
Labels are never invented: -1 = negative, 0 = neutral, 1 = positive, as given in the data.
"""
import pandas as pd

from src.config import DATA_FILE, LABELS
from src.preprocess import preprocess

SCORE_TO_LABEL = {-1: "negative", 0: "neutral", 1: "positive"}
TEXT_COLUMN_NAMES = ["comment", "text", "review", "feedback", "sentence"]


def _read_file(source, name=None):
    name = str(name or getattr(source, "name", source)).lower()
    return pd.read_excel(source) if name.endswith((".xlsx", ".xls")) else pd.read_csv(source)


def _scores_to_labels(scores):
    """Map -1/0/1 to names. Stops with an error on any other value instead of guessing."""
    known = scores.dropna()
    bad = sorted(set(known[~known.isin(list(SCORE_TO_LABEL))]))
    if bad:
        raise ValueError(f"Unexpected sentiment values {bad}; expected -1, 0 or 1.")
    return scores.map(lambda s: SCORE_TO_LABEL[int(s)] if pd.notna(s) else None)


def _reshape_wide(df):
    """Kaggle layout -> one row per comment (text, category, sentiment)."""
    cols = list(df.columns)
    if len(cols) % 2 != 0:
        raise ValueError(f"Expected pairs of (label, text) columns but found {len(cols)} columns: {cols}")
    parts = []
    for i in range(0, len(cols), 2):
        label_col, text_col = cols[i], cols[i + 1]
        is_pair = pd.api.types.is_numeric_dtype(df[label_col]) and not pd.api.types.is_numeric_dtype(df[text_col])
        if not is_pair:
            raise ValueError(f"'{label_col}' / '{text_col}' is not a (numeric label, text) pair. Columns found: {cols}")
        parts.append(pd.DataFrame({
            "text": df[text_col],
            "category": str(label_col).strip().lower(),
            "sentiment": _scores_to_labels(df[label_col]),
        }))
    return pd.concat(parts, ignore_index=True)


def load_labelled(source=DATA_FILE, name=None):
    """Read the labelled dataset (either layout) -> DataFrame[text, category, sentiment]."""
    df = _read_file(source, name)
    lower = {str(c).strip().lower(): c for c in df.columns}

    if "text" in lower and ("sentiment" in lower or "sentiment_score" in lower):   # Hugging Face layout
        out = pd.DataFrame({"text": df[lower["text"]]})
        out["category"] = df[lower["category"]] if "category" in lower else "unknown"
        if "sentiment" in lower:
            labels = df[lower["sentiment"]].astype(object).where(df[lower["sentiment"]].notna(), None)
            labels = labels.map(lambda s: str(s).strip().lower() if s is not None else None)
            bad = sorted(set(labels.dropna()) - set(LABELS))
            if bad:
                raise ValueError(f"Unexpected sentiment labels {bad}; expected {LABELS}.")
            out["sentiment"] = labels
        else:
            out["sentiment"] = _scores_to_labels(df[lower["sentiment_score"]])
        return out

    return _reshape_wide(df)                                                         # Kaggle layout


def clean_labelled(df):
    """Handle missing values, empty text, conflicting labels and duplicates.

    Returns (clean_df, report). The report records every row removed and why.
    Duplicates are removed BEFORE the train/test split so that the same comment
    can never appear in both sets (otherwise accuracy would be inflated).
    """
    report = {"rows_loaded": len(df)}
    df = df.copy()

    n = len(df)
    df = df.dropna(subset=["sentiment"])
    report["missing_label_removed"] = n - len(df)

    n = len(df)
    df = df.dropna(subset=["text"])
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"] != ""]
    report["missing_text_removed"] = n - len(df)

    df["clean"] = df["text"].apply(preprocess)
    n = len(df)
    df = df[df["clean"] != ""]
    report["empty_after_preprocessing_removed"] = n - len(df)
    report["class_counts_labelled"] = df["sentiment"].value_counts().reindex(LABELS, fill_value=0).to_dict()

    # Same comment with different labels = ambiguous. Remove it rather than guess.
    n_labels = df.groupby("clean")["sentiment"].nunique()
    conflicting = n_labels[n_labels > 1].index
    report["conflicting_texts_removed"] = int(len(conflicting))
    df = df[~df["clean"].isin(conflicting)]

    n = len(df)
    df = df.drop_duplicates("clean")
    report["duplicates_removed"] = n - len(df)

    df = df.reset_index(drop=True)
    report["rows_kept"] = len(df)
    report["class_counts_kept"] = df["sentiment"].value_counts().reindex(LABELS, fill_value=0).to_dict()
    return df, report


def prepare(source=DATA_FILE, name=None):
    """Load + clean in one step -> (clean_df, report)."""
    return clean_labelled(load_labelled(source, name))


def load_unlabelled(source, name=None):
    """Read a file of NEW comments to analyse. Needs one text column (name is detected)."""
    df = _read_file(source, name)
    lower = {str(c).strip().lower(): c for c in df.columns}
    for candidate in TEXT_COLUMN_NAMES:
        if candidate in lower:
            df = df.rename(columns={lower[candidate]: "text"})
            break
    else:
        raise ValueError(f"No text column found. Expected one of {TEXT_COLUMN_NAMES}; the file has {list(df.columns)}.")
    df = df.dropna(subset=["text"])
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"] != ""]
    if df.empty:
        raise ValueError("The file has no comments.")
    return df.reset_index(drop=True)
