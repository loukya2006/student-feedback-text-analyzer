"""Automated tests.  Run from the project folder:  python tests\\test_pipeline.py"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd

from src import config, dataset, model
from src.analyzer import run_analysis
from src.preprocess import preprocess


class TestPreprocess(unittest.TestCase):
    def test_stopwords_removed(self):
        self.assertNotIn("the", preprocess("The lab is good").split())

    def test_negation_kept(self):
        words = preprocess("The teaching is not good").split()
        self.assertIn("not", words)
        self.assertIn("good", words)


class TestDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = dataset.load_labelled(config.DATA_FILE)
        cls.clean, cls.report = dataset.clean_labelled(cls.raw)

    def test_reshape(self):
        self.assertEqual(list(self.raw.columns), ["text", "category", "sentiment"])
        self.assertEqual(len(self.raw), 1110)               # 185 rows x 6 feedback areas
        self.assertEqual(self.raw["category"].nunique(), 6)

    def test_labels_are_the_three_classes(self):
        self.assertEqual(set(self.clean["sentiment"]), set(config.LABELS))

    def test_cleaning(self):
        self.assertEqual(self.clean["clean"].duplicated().sum(), 0)
        self.assertEqual(self.clean["sentiment"].isna().sum(), 0)
        self.assertEqual(self.report["rows_kept"], len(self.clean))

    def test_huggingface_layout(self):
        path = os.path.join(tempfile.gettempdir(), "hf_layout.csv")
        d = self.raw.rename(columns={"sentiment": "sentiment"}).copy()
        d["sentiment_score"] = d["sentiment"].map({"negative": -1, "neutral": 0, "positive": 1})
        d.to_csv(path, index=False)
        loaded = dataset.load_labelled(path)
        self.assertEqual(len(loaded), len(self.raw))
        os.remove(path)

    def test_bad_file_is_rejected(self):
        path = os.path.join(tempfile.gettempdir(), "bad.csv")
        pd.DataFrame({"a": [1], "b": ["x"], "c": ["y"]}).to_csv(path, index=False)
        with self.assertRaises(ValueError):
            dataset.load_labelled(path)
        os.remove(path)

    def test_unlabelled_upload(self):
        path = os.path.join(tempfile.gettempdir(), "upload.csv")
        pd.DataFrame({"Comment": ["good lab", "", "bad books"]}).to_csv(path, index=False)
        self.assertEqual(len(dataset.load_unlabelled(path)), 2)
        os.remove(path)
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
            f.write("other\nhello\n")
        with self.assertRaises(ValueError):
            dataset.load_unlabelled(f.name)
        os.remove(f.name)


class TestModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.clean, cls.report = dataset.prepare(config.DATA_FILE)
        cls.sm, cls.test_df = model.train(cls.clean, cls.report)

    def test_split_sizes(self):
        m = self.sm.metrics
        self.assertEqual(m["n_train"] + m["n_test"], len(self.clean))

    def test_no_comment_in_both_train_and_test(self):
        train_part = self.clean.loc[~self.clean.index.isin(self.test_df.index)]
        self.assertEqual(len(train_part), self.sm.metrics["n_train"])
        self.assertTrue(set(train_part["clean"]).isdisjoint(set(self.test_df["clean"])))   # no leakage

    def test_confusion_matrix_covers_test_set(self):
        m = self.sm.metrics
        self.assertEqual(sum(sum(row) for row in m["confusion_matrix"]), m["n_test"])
        self.assertEqual(m["labels"], config.LABELS)

    def test_accuracy_is_reasonable(self):
        m = self.sm.metrics
        self.assertGreater(m["test_accuracy"], 0.5)
        self.assertGreater(m["train_accuracy"], m["test_accuracy"])

    def test_predictions_are_valid_classes(self):
        preds = self.sm.predict(["excellent teaching", "books are not available", "ok"])
        self.assertTrue(set(preds) <= set(config.LABELS))

    def test_clear_sentences(self):
        self.assertEqual(self.sm.predict(["The lab computers are very slow and not working"])[0], "negative")
        self.assertEqual(self.sm.predict(["books are not available"])[0], "negative")
        self.assertEqual(self.sm.predict(["excellent and amazing"])[0], "positive")

    def test_save_and_load(self):
        old = (config.MODEL_FILE, config.VECTORIZER_FILE, config.METRICS_FILE, config.TEST_SET_FILE, config.MODEL_DIR)
        tmp = tempfile.mkdtemp()
        try:
            config.MODEL_DIR = tmp
            config.MODEL_FILE = os.path.join(tmp, "model.pkl")
            config.VECTORIZER_FILE = os.path.join(tmp, "vectorizer.pkl")
            config.METRICS_FILE = os.path.join(tmp, "metrics.json")
            config.TEST_SET_FILE = os.path.join(tmp, "test_set.csv")
            model.save(self.sm, self.test_df)
            loaded = model.load()
            text = ["the library has very few books"]
            self.assertEqual(loaded.predict(text), self.sm.predict(text))
        finally:
            (config.MODEL_FILE, config.VECTORIZER_FILE, config.METRICS_FILE, config.TEST_SET_FILE, config.MODEL_DIR) = old


class TestAnalyzer(unittest.TestCase):
    def test_dashboard_numbers_come_from_the_model(self):
        clean, report = dataset.prepare(config.DATA_FILE)
        sm, test_df = model.train(clean, report)
        r = run_analysis(test_df, sm)
        independent = pd.Series(sm.predict(test_df["text"])).value_counts().reindex(config.LABELS, fill_value=0)
        self.assertEqual(list(r["counts"]), list(independent))        # same as calling the model directly
        self.assertEqual(int(r["counts"].sum()), len(test_df))
        self.assertAlmostEqual(float(r["percent"].sum()), 100, delta=0.5)
        self.assertGreater(len(r["top_words"]), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
