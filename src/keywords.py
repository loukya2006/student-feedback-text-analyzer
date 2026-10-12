"""Step 5 - Frequent words (what the notebook shows with word clouds)."""
from collections import Counter


def top_words(processed_texts, n=10):
    """Most frequent words in already-preprocessed comments -> [(word, count), ...]."""
    counter = Counter()
    for text in processed_texts:
        counter.update(w for w in text.split() if len(w) > 2)
    return counter.most_common(n)
