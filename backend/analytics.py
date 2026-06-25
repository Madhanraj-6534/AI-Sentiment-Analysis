"""
analytics.py
Computes analytics and builds Plotly chart figures for the dashboard.
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


DARK_BLUE = "#0B1F3A"
ACCENT_BLUE = "#1565C0"
POSITIVE_COLOR = "#2ECC71"
NEGATIVE_COLOR = "#E74C3C"
PAPER_BG = "rgba(0,0,0,0)"
FONT_COLOR = "#E6EDF7"


def build_history_summary(history_df: pd.DataFrame) -> dict:
    """
    Computes summary KPI numbers from prediction history.

    Args:
        history_df: DataFrame with a 'prediction' column.

    Returns:
        dict with total, positive, negative, positive_pct, negative_pct
    """
    if history_df is None or history_df.empty:
        return {"total": 0, "positive": 0, "negative": 0, "positive_pct": 0.0, "negative_pct": 0.0}

    total = len(history_df)
    positive = int((history_df["prediction"] == "Positive").sum())
    negative = int((history_df["prediction"] == "Negative").sum())
    positive_pct = round((positive / total) * 100, 1) if total else 0.0
    negative_pct = round((negative / total) * 100, 1) if total else 0.0

    return {
        "total": total,
        "positive": positive,
        "negative": negative,
        "positive_pct": positive_pct,
        "negative_pct": negative_pct,
    }


def build_pie_chart(positive: int, negative: int) -> go.Figure:
    """Builds a pie chart showing sentiment distribution."""
    if positive == 0 and negative == 0:
        positive, negative = 1, 1  # avoid an empty chart; placeholder only

    fig = go.Figure(
        data=[
            go.Pie(
                labels=["Positive", "Negative"],
                values=[positive, negative],
                hole=0.45,
                marker=dict(colors=[POSITIVE_COLOR, NEGATIVE_COLOR]),
                textinfo="label+percent",
                textfont=dict(color="white", size=14),
            )
        ]
    )
    fig.update_layout(
        title="Sentiment Distribution",
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PAPER_BG,
        font=dict(color=FONT_COLOR),
        legend=dict(orientation="h", y=-0.1),
        margin=dict(t=60, b=40, l=20, r=20),
    )
    return fig


def build_bar_chart(positive: int, negative: int) -> go.Figure:
    """Builds a bar chart comparing positive vs negative counts."""
    fig = go.Figure(
        data=[
            go.Bar(
                x=["Positive", "Negative"],
                y=[positive, negative],
                marker_color=[POSITIVE_COLOR, NEGATIVE_COLOR],
                text=[positive, negative],
                textposition="outside",
            )
        ]
    )
    fig.update_layout(
        title="Prediction Counts",
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PAPER_BG,
        font=dict(color=FONT_COLOR),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)"),
        margin=dict(t=60, b=40, l=20, r=20),
    )
    return fig


def build_confusion_matrix_chart(cm: list, labels: list) -> go.Figure:
    """
    Builds a heatmap visualization of the confusion matrix.

    Args:
        cm: 2D list, confusion matrix values.
        labels: list of class labels matching matrix order.
    """
    fig = px.imshow(
        cm,
        text_auto=True,
        x=labels,
        y=labels,
        color_continuous_scale="Blues",
        labels=dict(x="Predicted", y="Actual", color="Count"),
    )
    fig.update_layout(
        title="Confusion Matrix",
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PAPER_BG,
        font=dict(color=FONT_COLOR),
        margin=dict(t=60, b=40, l=20, r=20),
    )
    return fig


def build_history_trend_chart(history_df: pd.DataFrame) -> go.Figure:
    """Builds a simple trend line of predictions over time (cumulative)."""
    if history_df is None or history_df.empty:
        fig = go.Figure()
        fig.update_layout(
            title="No history available yet",
            paper_bgcolor=PAPER_BG,
            plot_bgcolor=PAPER_BG,
            font=dict(color=FONT_COLOR),
        )
        return fig

    df = history_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values("timestamp")
    df["positive_count"] = (df["prediction"] == "Positive").cumsum()
    df["negative_count"] = (df["prediction"] == "Negative").cumsum()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["positive_count"],
        mode="lines", name="Positive (cumulative)",
        line=dict(color=POSITIVE_COLOR, width=3),
    ))
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["negative_count"],
        mode="lines", name="Negative (cumulative)",
        line=dict(color=NEGATIVE_COLOR, width=3),
    ))
    fig.update_layout(
        title="Prediction Trend Over Time",
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PAPER_BG,
        font=dict(color=FONT_COLOR),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)"),
        legend=dict(orientation="h", y=-0.2),
        margin=dict(t=60, b=40, l=20, r=20),
    )
    return fig
