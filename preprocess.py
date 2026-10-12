"""Step 1 - Text preprocessing.

Notebook (cells 7-11): remove_stopwords() splits a review into words and drops the
NLTK English stopwords. We do the same, with ONE change: negation words
(not, no, nor, never, don't, ...) are kept, because in student feedback
"not good" and "good" mean opposite things.
"""
import re


def _load_stopwords():
    """NLTK English stopwords (as in the notebook); scikit-learn's list is a fallback."""
    try:
        import nltk
        from nltk.corpus import stopwords
        try:
            words = stopwords.words("english")
        except LookupError:
            nltk.download("stopwords", quiet=True)
            words = stopwords.words("english")
        return set(words), "NLTK"
    except Exception:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        return set(ENGLISH_STOP_WORDS), "scikit-learn fallback (NLTK stopwords not available)"


def _is_negation(word):
    return word in {"no", "nor", "not", "never", "cannot"} or word.endswith("n't")


_ALL_STOPWORDS, STOPWORD_SOURCE = _load_stopwords()
STOP_WORDS = {w for w in _ALL_STOPWORDS if not _is_negation(w)}


def preprocess(text):
    """Lowercase, split into words, remove stopwords (keeping negations)."""
    text = str(text).lower().replace("\u2019", "'")
    words = re.findall(r"[a-z]+(?:'[a-z]+)?", text)
    return " ".join(w for w in words if w not in STOP_WORDS)
