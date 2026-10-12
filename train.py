"""Train and evaluate the sentiment model.

Run:  python train.py
      python train.py --data path\to\file.xlsx     (Kaggle xlsx or Hugging Face csv)
"""
import argparse
import os
import sys

from src import config, dataset, model, visualize
from src.preprocess import STOPWORD_SOURCE


def main():
    parser = argparse.ArgumentParser(description="Train the student feedback sentiment model")
    parser.add_argument("--data", default=config.DATA_FILE, help="labelled dataset (xlsx or csv)")
    args = parser.parse_args()
    if not os.path.exists(args.data):
        sys.exit(f"File not found: {args.data}")

    # ---- 1. Load and clean ----
    raw = dataset.load_labelled(args.data)
    print(f"Dataset file : {args.data}")
    print(f"Loaded       : {len(raw)} comments, columns {list(raw.columns)}")
    print(f"Stopword list: {STOPWORD_SOURCE}")
    clean_df, report = dataset.clean_labelled(raw)

    print("\n=== DATA CLEANING ===")
    for key, value in report.items():
        print(f"{key.replace('_', ' '):<36}: {value}")

    # ---- 2. Train + evaluate ----
    sentiment_model, test_df = model.train(clean_df, report)
    model.save(sentiment_model, test_df)
    m = sentiment_model.metrics

    print("\n=== TRAIN / TEST SPLIT ===")
    print(f"Training comments : {m['n_train']}  {m['train_class_counts']}")
    print(f"Test comments     : {m['n_test']}  {m['test_class_counts']}  (never used for learning)")

    print("\n=== EVALUATION (on the unseen test comments) ===")
    print(f"Test accuracy     : {m['test_accuracy']:.1%}")
    print(f"Training accuracy : {m['train_accuracy']:.1%}   (higher, because the model has seen these)")
    print(f"Baseline accuracy : {m['baseline_accuracy']:.1%}   (always guessing '{m['baseline_class']}')")
    print("\nConfusion matrix (rows = actual, columns = predicted):")
    print(f"{'':>10}" + "".join(f"{l:>10}" for l in m["labels"]))
    for label, row in zip(m["labels"], m["confusion_matrix"]):
        print(f"{label:>10}" + "".join(f"{v:>10}" for v in row))
    print("\nRecall per class (correct / actual, from the confusion matrix):")
    for label, r in m["recall_per_class"].items():
        print(f"  {label:<9}: {'n/a' if r is None else f'{r:.0%}'}")

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    visualize.confusion_matrix_chart(m["confusion_matrix"], m["labels"], config.OUTPUT_DIR)
    clean_df.to_csv(os.path.join(config.OUTPUT_DIR, "cleaned_dataset.csv"), index=False)
    print(f"\nSaved model files to '{config.MODEL_DIR}', confusion_matrix.png and cleaned_dataset.csv to '{config.OUTPUT_DIR}'.")


if __name__ == "__main__":
    main()
