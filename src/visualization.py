"""Plotly figures (TRD section 9). One fixed colour + marker per activity across all plots."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from .data_loader import ACTIVITY_NAMES

# Okabe-Ito colour-blind-safe palette + distinct marker symbols (colour is never the only cue)
ACTIVITY_COLORS = {
    1: "#0072B2",
    2: "#56B4E9",
    3: "#009E73",
    4: "#E69F00",
    5: "#D55E00",
    6: "#CC79A7",
}
ACTIVITY_SYMBOLS = {1: "circle", 2: "diamond", 3: "square", 4: "triangle-up", 5: "x", 6: "cross"}
METRIC_COLORS = {"Accuracy": "#0072B2", "Macro precision": "#009E73", "Macro recall": "#E69F00", "Macro F1": "#CC79A7"}


def embedding_figure(emb: np.ndarray, y: np.ndarray, title: str, axis_prefix: str) -> go.Figure:
    fig = go.Figure()
    for label, name in ACTIVITY_NAMES.items():
        mask = y == label
        if not mask.any():
            continue
        fig.add_trace(
            go.Scattergl(
                x=emb[mask, 0],
                y=emb[mask, 1],
                mode="markers",
                name=name,
                marker={
                    "size": 6,
                    "opacity": 0.75,
                    "color": ACTIVITY_COLORS[label],
                    "symbol": ACTIVITY_SYMBOLS[label],
                },
                hovertemplate=f"{name}<br>{axis_prefix} 1=%{{x:.3f}}<br>{axis_prefix} 2=%{{y:.3f}}<extra></extra>",
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title=f"{axis_prefix} 1",
        yaxis_title=f"{axis_prefix} 2",
        legend_title_text="Activity",
        height=520,
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
    )
    return fig


def explained_variance_figure(evr: np.ndarray, cumulative: np.ndarray) -> go.Figure:
    idx = np.arange(1, len(evr) + 1)
    fig = go.Figure()
    fig.add_bar(x=idx, y=evr, name="Per component", marker_color="#0072B2")
    fig.add_scatter(x=idx, y=cumulative, name="Cumulative", mode="lines+markers", marker_color="#D55E00")
    fig.update_layout(
        title="PCA explained variance ratio (selected components)",
        xaxis_title="Principal component",
        yaxis_title="Explained variance ratio",
        yaxis_range=[0, 1.02],
        height=380,
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
    )
    return fig


def full_cumulative_figure(cumulative: np.ndarray, n_selected: int) -> go.Figure:
    idx = np.arange(1, len(cumulative) + 1)
    fig = go.Figure()
    fig.add_scatter(x=idx, y=cumulative, mode="lines", name="Cumulative", line={"color": "#0072B2"})
    for thr, dash in ((0.90, "dot"), (0.95, "dash"), (0.99, "dashdot")):
        fig.add_hline(y=thr, line_dash=dash, line_color="#888", annotation_text=f"{int(thr * 100)}%")
    fig.add_vline(x=n_selected, line_color="#D55E00", annotation_text=f"selected k={n_selected}")
    fig.update_layout(
        title="PCA cumulative explained variance - all components",
        xaxis_title="Number of components",
        yaxis_title="Cumulative explained variance",
        yaxis_range=[0, 1.02],
        xaxis_type="log",
        height=380,
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
    )
    return fig


def singular_values_figure(sv: np.ndarray) -> go.Figure:
    idx = np.arange(1, len(sv) + 1)
    fig = go.Figure(go.Bar(x=idx, y=sv, marker_color="#009E73"))
    fig.update_layout(
        title="Truncated SVD singular values",
        xaxis_title="Component",
        yaxis_title="Singular value",
        height=380,
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
    )
    return fig


def metric_comparison_figure(rows: list[dict]) -> go.Figure:
    """rows: dicts with keys Model, Accuracy, 'Macro precision', 'Macro recall', 'Macro F1'."""
    fig = go.Figure()
    for metric, color in METRIC_COLORS.items():
        fig.add_bar(name=metric, x=[r["Model"] for r in rows], y=[r[metric] for r in rows], marker_color=color)
    fig.update_layout(
        barmode="group",
        title="Test-set metrics (same test samples for every model)",
        yaxis_title="Score",
        yaxis_range=[0, 1.05],
        height=420,
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
    )
    return fig


def confusion_figure(cm: np.ndarray, title: str) -> go.Figure:
    names = [ACTIVITY_NAMES[k] for k in sorted(ACTIVITY_NAMES)]
    fig = go.Figure(
        go.Heatmap(
            z=cm,
            x=names,
            y=names,
            text=cm,
            texttemplate="%{text}",
            colorscale="Blues",
            showscale=False,
            hovertemplate="True %{y}<br>Predicted %{x}<br>Count %{z}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Predicted activity",
        yaxis_title="True activity",
        yaxis_autorange="reversed",
        height=480,
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
    )
    return fig


def class_distribution_figure(df) -> go.Figure:
    fig = go.Figure()
    fig.add_bar(x=df["Activity"], y=df["Train"], name="Train", marker_color="#0072B2")
    fig.add_bar(x=df["Activity"], y=df["Test"], name="Test", marker_color="#E69F00")
    fig.update_layout(barmode="group", height=320, yaxis_title="Samples", margin={"l": 10, "r": 10, "t": 30, "b": 10})
    return fig
