"""Plotly grafikleri. Sadece sade, standart Plotly çağrıları kullanılır."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

MAX_POINTS = 30_000   # tarayıcıyı yormamak için çok uzun seriler seyreltilir (yalnızca çizimde)


def _thin(n: int) -> int:
    return max(1, int(np.ceil(n / MAX_POINTS)))


def series_plot(series, scores=None, threshold=None, train_end=None, val_end=None, title=None):
    """Zaman serisi + gerçek anomali pencereleri (kırmızı bant) + (varsa) skor ve alarmlar."""
    df = series.df
    step = _thin(len(df))
    d = df.iloc[::step]
    two = scores is not None
    fig = make_subplots(rows=2 if two else 1, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        row_heights=[0.65, 0.35] if two else [1.0])
    fig.add_trace(go.Scatter(x=d["timestamp"], y=d["value"], mode="lines", name="değer",
                             line=dict(width=1)), row=1, col=1)
    for a, b in series.windows:
        fig.add_vrect(x0=a, x1=b, fillcolor="red", opacity=0.15, line_width=0, row=1, col=1)
    if series.windows:   # kırmızı bantlar legend'da görünsün diye boş bir iz ekliyoruz
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name="gerçek anomali penceresi (NAB)",
                                 marker=dict(symbol="square", size=12, color="rgba(255,0,0,0.3)")), row=1, col=1)
    if two:
        sc = np.asarray(scores, dtype=float)
        fig.add_trace(go.Scatter(x=d["timestamp"], y=sc[::step], mode="lines", name="anomali skoru",
                                 line=dict(width=1)), row=2, col=1)
        if threshold is not None:
            fig.add_trace(go.Scatter(x=[d["timestamp"].iloc[0], d["timestamp"].iloc[-1]],
                                     y=[threshold, threshold], mode="lines", name="eşik",
                                     line=dict(dash="dash", width=1)), row=2, col=1)
            alert = sc > threshold
            fig.add_trace(go.Scatter(x=df["timestamp"][alert], y=df["value"][alert], mode="markers",
                                     name="alarm", marker=dict(size=4, color="crimson")), row=1, col=1)
    for idx in (train_end, val_end):
        if idx is not None and 0 < idx < len(df):
            x = df["timestamp"].iloc[idx]
            fig.add_shape(type="line", x0=x, x1=x, y0=0, y1=1, xref="x", yref="paper",
                          line=dict(dash="dot", width=1, color="gray"))
    fig.update_layout(title=title or series.name, height=520 if two else 380,
                      margin=dict(l=40, r=20, t=50, b=30), legend=dict(orientation="h"))
    return fig


def metric_bars(df: pd.DataFrame, metrics: list[str], title: str = "", names: dict | None = None):
    """Model başına ortalama ölçüler (her ölçü ayrı panel). names: {teknik ad: okunur ad}."""
    names = names or {}
    m = df.groupby("model", sort=False)[metrics].mean().reset_index().melt(
        id_vars="model", var_name="ölçü", value_name="değer")
    m["ölçü"] = m["ölçü"].map(lambda k: names.get(k, k))
    fig = px.bar(m, x="model", y="değer", color="model", facet_col="ölçü", title=title)
    fig.update_yaxes(matches=None, showticklabels=True)
    fig.update_layout(showlegend=False, height=380)
    return fig


def line_by_level(df: pd.DataFrame, x: str, y: str, title: str = "", y_label: str | None = None,
                  x_label: str | None = None):
    """x (ör. gürültü seviyesi) boyunca modellerin ortalama y değeri."""
    g = df.groupby(["model", x], sort=False)[y].mean().reset_index()
    fig = px.line(g, x=x, y=y, color="model", markers=True, title=title)
    fig.update_yaxes(title=y_label or y)
    fig.update_xaxes(title=x_label or x)
    fig.update_layout(height=380)
    return fig


def loss_curve(history: list[float], title: str = "Autoencoder eğitim kaybı (MSE)"):
    fig = go.Figure(go.Scatter(x=list(range(1, len(history) + 1)), y=history, mode="lines+markers"))
    fig.update_layout(title=title, xaxis_title="epoch", yaxis_title="kayıp", height=300)
    return fig
