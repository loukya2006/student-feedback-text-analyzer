"""Interactive dashboard.  Run with:  streamlit run app.py

The dashboard only adds a user interface. Every sentiment it shows is predicted by the
trained Logistic Regression model (src/model.py) - nothing is hard-coded.
"""
import pandas as pd
import streamlit as st

from src import config, dataset, model, visualize
from src.analyzer import run_analysis

st.set_page_config(page_title="Student Feedback Analyzer", layout="wide")


@st.cache_resource
def get_model():
    return model.load_or_train()      # loads the saved model, or trains it if it is missing


sm = get_model()
m = sm.metrics

st.title("Student Feedback Text Analyzer")
st.caption(f"TF-IDF features + Logistic Regression | stopwords: {m['stopword_source']}")

# ---------- Sidebar: choose the comments to analyse ----------
st.sidebar.header("Comments to analyse")
uploaded = st.sidebar.file_uploader("Upload your own CSV/Excel (needs a 'comment' or 'text' column)", type=["csv", "xlsx"])
try:
    if uploaded is not None:
        df = dataset.load_unlabelled(uploaded, name=uploaded.name)
        source = uploaded.name
        st.sidebar.warning("If this file contains comments the model was trained on, the results are not an honest test.")
    else:
        df = pd.read_csv(config.TEST_SET_FILE)
        source = "held-out test comments"
        st.sidebar.info("Using the test comments the model never saw during training.")
except Exception as err:
    st.error(f"Could not read the file: {err}")
    st.stop()

# ---------- Run the real pipeline: preprocess -> model prediction ----------
r = run_analysis(df, sm)
charts = visualize.save_all(r, m, config.OUTPUT_DIR)

st.subheader(f"Predicted sentiment of {len(r['df'])} comments ({source})")
c1, c2, c3 = st.columns(3)
for col, label in zip([c1, c2, c3], config.LABELS):
    col.metric(label.capitalize(), f"{r['percent'][label]}%", f"{r['counts'][label]} comments", delta_color="off")

tab1, tab2, tab3, tab4 = st.tabs(["Sentiment", "Frequent words", "Model evaluation", "Try a comment"])

with tab1:
    left, right = st.columns([1, 2])
    left.image(charts["sentiment"])
    show = right.multiselect("Filter comments by predicted sentiment", config.LABELS, default=config.LABELS)
    cols = [c for c in ["text", "category", "predicted", "sentiment"] if c in r["df"].columns]
    right.dataframe(r["df"][r["df"]["predicted"].isin(show)][cols], height=380)
    if r["actual_counts"] is not None:
        st.write("**Predicted vs actual counts** (actual labels come from the dataset):")
        st.dataframe(pd.DataFrame({"predicted": r["counts"], "actual": r["actual_counts"]}).T)
    if r["by_category"] is not None:
        st.write("**Predicted sentiment by category**")
        st.dataframe(r["by_category"])

with tab2:
    st.image(charts["words"])
    st.write("**Most frequent words:** " + ", ".join(f"{w} ({c})" for w, c in r["top_words"]))
    if "wordcloud" in charts:
        st.image(charts["wordcloud"])

with tab3:
    st.write(f"The data was split before training: the model learned from **{m['n_train']}** comments "
             f"and was then tested on **{m['n_test']}** comments it had never seen.")
    a, b, c = st.columns(3)
    a.metric("Test accuracy (unseen comments)", f"{m['test_accuracy']:.1%}")
    b.metric("Training accuracy (seen comments)", f"{m['train_accuracy']:.1%}")
    c.metric(f"Baseline: always '{m['baseline_class']}'", f"{m['baseline_accuracy']:.1%}")
    st.caption("Training accuracy is higher because the model has already seen those comments. "
               "Only the test accuracy shows how well it handles new feedback.")
    left, right = st.columns(2)
    left.image(charts["confusion"])
    right.write("**Recall per class** (from the confusion matrix)")
    right.dataframe(pd.DataFrame({"recall": m["recall_per_class"]}))
    right.write("**Dataset cleaning**")
    right.dataframe(pd.DataFrame({"value": {k.replace("_", " "): str(v) for k, v in m["cleaning"].items()}}))

with tab4:
    text = st.text_area("Type a student comment", "The library has very few books and they are not available.")
    if st.button("Predict"):
        label, chances = sm.predict_one(text)
        st.write(f"**Predicted sentiment:** {label}")
        st.write("Probabilities: " + ", ".join(f"{k} {v:.0%}" for k, v in chances.items()))
