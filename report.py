"""Builds one self-contained HTML report (charts embedded)."""
import base64
import html
from datetime import datetime

from src.config import LABELS


def _img(path):
    with open(path, "rb") as f:
        return f'<img src="data:image/png;base64,{base64.b64encode(f.read()).decode()}" style="max-width:100%">'


def build_html(analysis, metrics, chart_paths, out_path, source_name="feedback"):
    e = html.escape
    df, counts, pct = analysis["df"], analysis["counts"], analysis["percent"]
    c = metrics["cleaning"]
    p = [f"""<html><head><meta charset="utf-8"><title>Student Feedback Report</title>
<style>body{{font-family:Arial,sans-serif;max-width:900px;margin:30px auto;color:#222}}
h1{{color:#1f4e79}} h2{{border-bottom:2px solid #ddd;padding-bottom:4px;margin-top:30px}}
table{{border-collapse:collapse}} td,th{{border:1px solid #ccc;padding:6px 12px;text-align:left}}
th{{background:#f0f4f8}}</style></head><body>
<h1>Student Feedback Analysis Report</h1>
<p>Source: {e(str(source_name))} | Comments analysed: {len(df)} | Generated: {datetime.now():%Y-%m-%d %H:%M}<br>
Sentiment is predicted by a Logistic Regression model trained on TF-IDF features.</p>
<h2>Predicted sentiment</h2><table><tr><th>Sentiment</th><th>Comments</th><th>Share</th></tr>"""]
    for label in LABELS:
        p.append(f"<tr><td>{label}</td><td>{counts.get(label, 0)}</td><td>{pct.get(label, 0)}%</td></tr>")
    p.append("</table>" + _img(chart_paths["sentiment"]))

    p.append("<h2>Frequent words</h2><p>" + ", ".join(f"{e(w)} ({n})" for w, n in analysis["top_words"]) + "</p>")
    p.append(_img(chart_paths["words"]))

    p.append("<h2>Model evaluation (held-out test set)</h2><table>")
    p.append(f"<tr><td>Test accuracy</td><td>{metrics['test_accuracy']:.1%} on {metrics['n_test']} unseen comments</td></tr>")
    p.append(f"<tr><td>Training accuracy</td><td>{metrics['train_accuracy']:.1%} on {metrics['n_train']} comments the model learned from</td></tr>")
    p.append(f"<tr><td>Baseline (always guess '{metrics['baseline_class']}')</td><td>{metrics['baseline_accuracy']:.1%}</td></tr>")
    for label, r in metrics["recall_per_class"].items():
        p.append(f"<tr><td>Recall: {label}</td><td>{'n/a' if r is None else f'{r:.0%}'}</td></tr>")
    p.append("</table>" + _img(chart_paths["confusion"]))

    p.append("<h2>Dataset cleaning</h2><table>")
    for key, value in c.items():
        p.append(f"<tr><td>{e(key.replace('_', ' '))}</td><td>{e(str(value))}</td></tr>")
    p.append("</table></body></html>")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("".join(p))
    return out_path
