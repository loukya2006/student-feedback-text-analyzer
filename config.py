"""All settings in one place (paths and model parameters)."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(ROOT, "data", "finalDataset0_2.xlsx")
MODEL_DIR = os.path.join(ROOT, "models")
OUTPUT_DIR = os.path.join(ROOT, "outputs")

MODEL_FILE = os.path.join(MODEL_DIR, "model.pkl")
VECTORIZER_FILE = os.path.join(MODEL_DIR, "vectorizer.pkl")
METRICS_FILE = os.path.join(MODEL_DIR, "metrics.json")
TEST_SET_FILE = os.path.join(MODEL_DIR, "test_set.csv")   # held-out comments, never used for training

LABELS = ["negative", "neutral", "positive"]

# Model settings (same ideas as the professor's notebook)
MAX_FEATURES = 2500        # TfidfVectorizer(max_features=2500) in the notebook
TEST_SIZE = 0.2            # notebook used 0.15
RANDOM_STATE = 42          # makes the split repeatable
CLASS_WEIGHT = "balanced"  # handles class imbalance (still plain Logistic Regression)
