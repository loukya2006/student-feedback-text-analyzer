# Student Feedback Text Analyzer

Classifies student feedback as **positive, neutral or negative** using only the techniques from the professor's
`MovieReviewAnalysis.ipynb`: stopword removal, **TF-IDF**, **Logistic Regression**, a **train/test split**, and
**accuracy + confusion matrix**. A dashboard shows the predicted sentiment counts, percentages, a chart and frequent words.

## Windows setup (run once)
Install Python 3.10 or newer from python.org (tick "Add Python to PATH"). In VS Code choose File > Open Folder
(the folder that contains `main.py`), open a terminal (Ctrl + `) and run:

```
py -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('stopwords')"
```
- PowerShell blocks activation? Run `Set-ExecutionPolicy -Scope Process Bypass`, then `venv\Scripts\Activate.ps1` (or use the Command Prompt terminal).
- The NLTK line downloads the stopword list once (internet needed), exactly as in the notebook.
  If it is missing, the program prints "scikit-learn fallback" and still runs.
- Optional word cloud: `pip install -r requirements-optional.txt`

## Run
| Step | Command |
|---|---|
| 1. Train and evaluate the model | `python train.py` |
| 2. Self-test (16 tests) | `python tests\test_pipeline.py` |
| 3. Terminal analysis (+ charts and `outputs\report.html`) | `python main.py` |
| 4. Dashboard (opens in the browser) | `streamlit run app.py` |
| Predict one comment | `python main.py --text "The lab computers are very slow"` |
| Analyse your own comments (CSV/Excel with a `comment` or `text` column) | `python main.py --input data\mycomments.csv` |
| Train on the Hugging Face CSV instead | `python train.py --data data\student_feedback_sentiment.csv` |

`python main.py` and the dashboard train the model automatically if step 1 was skipped.

## Dataset
`data/finalDataset0_2.xlsx` is the Kaggle student-feedback sheet: 185 students x 6 feedback areas
(teaching, coursecontent, examination, labwork, library_facilities, extracurricular). Each area has a numeric label
(-1 negative, 0 neutral, 1 positive) and a comment, so the program reshapes it into one row per comment (1,110 rows).
The Hugging Face version (`text, category, sentiment_score, sentiment`) is the same data already reshaped, and is also accepted.

**Cleaning (printed by `train.py`):** 5 rows without a label are removed; comments that become empty after stopword removal
are removed; 3 comments that appear with conflicting labels are removed (no guessing); duplicates are removed (525 rows, mostly
repeated "Good"), leaving 547 unique comments (305 positive, 121 neutral, 121 negative). Labels are never invented.

## How the train/test split protects the evaluation
The 547 comments are split **before** training: 80% (437) train the model, 20% (110) are put aside as a test set.
TF-IDF and Logistic Regression only ever see the training part. Accuracy and the confusion matrix are measured on the test part,
which the model has never seen, so the score reflects how it handles new feedback. Training accuracy (about 96%) is much higher
than test accuracy (about 65%) because the model has memorised what it was shown; only the test figure is the honest one.
Duplicates are removed first so the same comment cannot appear on both sides (leaving them in raised accuracy to about 83% in my test).
Because the data is imbalanced (56% positive), the dashboard also shows a baseline: always guessing "positive" is already about 56% accurate.
`class_weight="balanced"` is used so rarer classes (neutral, negative) are not ignored.

## Project structure
```
train.py                Trains + evaluates the model, saves it to models/
main.py                 Command-line analysis, report.html
app.py                  Streamlit dashboard (uses the saved model's predictions)
data/finalDataset0_2.xlsx   Labelled dataset
src/config.py           Paths and model settings
src/dataset.py          Load / reshape / clean the dataset
src/preprocess.py       Stopword removal (NLTK), negations kept
src/model.py            TF-IDF, Logistic Regression, split, evaluation, save/load
src/analyzer.py         Runs the model on comments -> counts, percentages, words
src/keywords.py         Frequent words
src/visualize.py        Pie chart, word chart, confusion matrix, optional word cloud
src/report.py           HTML report
tests/test_pipeline.py  Automated tests
models/                 Generated: model.pkl, vectorizer.pkl, metrics.json, test_set.csv
outputs/                Generated: charts, analyzed_feedback.csv, cleaned_dataset.csv, report.html
```

## Limitations
- Only 547 unique comments, many only one or two words, so accuracy is modest (about 65%, versus a 56% baseline) and neutral is the hardest class.
- Results change slightly with the stopword list (NLTK vs fallback) and the random split.
- Logistic Regression on TF-IDF reads words, not meaning, so sarcasm and mixed comments are often wrong, and "not good" is sometimes labelled neutral.
