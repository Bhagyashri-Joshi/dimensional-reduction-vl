"""Turn actual session results into tables, CSV and a Markdown report (PRD FR-21, FR-22).

Only values computed in the current session are used. Unavailable values are shown as N/A, never 0.
"""
from __future__ import annotations

import datetime as _dt

import pandas as pd

from .dimensionality_reduction import EmbeddingResult
from .evaluation import ClassificationResult


def fmt_params(params: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in params.items())


def embedding_run_key(r: EmbeddingResult) -> str:
    return f"{r.algorithm}|{sorted(r.params.items())}|n={r.n_samples}|seed={r.seed}|k={r.trust_k}"


def embedding_row(r: EmbeddingResult) -> dict:
    return {
        "Algorithm": r.algorithm,
        "Parameters": fmt_params(r.params),
        "Samples": r.n_samples,
        "Original dims": r.n_original,
        "Reduced dims": r.n_reduced,
        "Seed": r.seed,
        "Trustworthiness": f"{r.trustworthiness:.4f} (k={r.trust_k})" if r.trustworthiness is not None else "N/A",
        "Runtime (s)": round(r.runtime_s, 3),
    }


def embedding_table(history: dict, tsne: bool) -> pd.DataFrame:
    rows = [row for row in history.values() if (row["Algorithm"] == "t-SNE") == tsne]
    return pd.DataFrame(rows)


def comparison_rows(results: list[ClassificationResult], run_label: str | None = None) -> list[dict]:
    rows = []
    for r in results:
        row = {
            "Model": r.label,
            "Reduced dims": "561 (original)" if r.is_original else r.n_dims,
            "Accuracy": round(r.accuracy, 4),
            "Macro precision": round(r.precision_macro, 4),
            "Macro recall": round(r.recall_macro, 4),
            "Macro F1": round(r.f1_macro, 4),
            "Reduction time (s)": "N/A" if r.reduction_time_s is None else round(r.reduction_time_s, 3),
            "Classifier time (s)": round(r.classifier_time_s, 3),
            "Converged": "yes" if r.converged else "NO",
            "Settings": fmt_params(r.settings),
        }
        if run_label is not None:
            row = {"Run": run_label, **row}
        rows.append(row)
    return rows


def comparison_label(results: list[ClassificationResult]) -> str:
    non_base = next((r for r in results if not r.is_original), None)
    if non_base is None:
        return "baseline only"
    umap = next((r for r in results if r.method == "UMAP"), None)
    s = f"dims={non_base.n_dims}, seed={non_base.settings['seed']}"
    if umap:
        s += f", UMAP nn={umap.settings['n_neighbors']}, md={umap.settings['min_dist']}, fit={umap.settings['umap_fit_samples']}"
    return s


def comparison_history_table(history: dict) -> pd.DataFrame:
    rows: list[dict] = []
    for label, results in history.items():
        rows.extend(comparison_rows(results, label))
    return pd.DataFrame(rows)


def _md_table(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in df.itertuples(index=False):
        out.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(out)


def to_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def markdown_report(sim_history: dict, cmp_history: dict) -> str:
    now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "# Dimensionality Reduction Explorer - Experiment Report",
        f"_Generated {now} from the results computed in this session. No values are hard-coded._",
        "",
        "**Dataset:** UCI HAR (561 features, 6 activities, predefined train/test split)",
        "",
        "## Embedding runs (visualisation analysis)",
    ]
    for tsne in (False, True):
        df = embedding_table(sim_history, tsne)
        if not df.empty:
            lines += ["", "### t-SNE" if tsne else "### PCA / Truncated SVD / UMAP", "", _md_table(df)]
    if not sim_history:
        lines.append("\n_No embedding runs yet._")
    lines += ["", "## Classification comparison (Logistic Regression, predefined test set)"]
    df = comparison_history_table(cmp_history)
    lines += ["", _md_table(df) if not df.empty else "_No comparison runs yet._"]
    lines += [
        "",
        "## Methodology",
        "- Scaler and reducers fitted on the training split only; the test split was only transformed.",
        "- One Logistic Regression configuration (C=1.0, lbfgs, max_iter=1000) for all models.",
        "- t-SNE is evaluated separately (no out-of-sample transform in scikit-learn).",
        "- Single run per setting; differences between close scores may not be meaningful.",
    ]
    return "\n".join(lines)
