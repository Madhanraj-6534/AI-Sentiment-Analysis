"""
ui.py
Streamlit UI rendering functions for every page of the
AI-Powered Customer Review Sentiment Analysis System.

NOTE ON SCOPE:
This module is a pure presentation layer. It does not contain any
machine learning, database, or analytics logic — it only calls the
existing backend objects (predictor, database) and the existing
backend.analytics chart builders, then renders the results inside a
modern SaaS-style Streamlit UI (cards, badges, tabs, KPI grids, etc).

All public function names and signatures below are unchanged from the
previous version of this file, since app.py calls them directly:
    - load_css(css_path)
    - render_home_page(metrics, db_summary)
    - render_predict_page(predictor, database)
    - render_batch_page(predictor, database)
    - render_analytics_page(metrics, db_summary, history_df)
    - render_history_page(database)
"""

import os
import io
import time
import datetime as dt

import pandas as pd
import streamlit as st

from backend.predictor import PredictorError
from backend.database import DatabaseError
from backend import analytics as an


# ---------------------------------------------------------------------------
# App meta (frontend-only constants — no business logic)
# ---------------------------------------------------------------------------

APP_NAME = "SentimentAI"
APP_VERSION = "v2.0.0"
APP_TAGLINE = "Customer Review Intelligence"
DEVELOPER_NAME = "MADHAN RAJ P"
GITHUB_URL = "https://github.com/Madhanraj-6534/"

# Chart theme overrides applied on top of figures returned by backend.analytics.
# This only restyles colors/fonts on the Figure object already built by the
# backend — it does not touch any analytics computation logic.
_CHART_FONT_COLOR = "#0F172A"
_CHART_GRID_COLOR = "rgba(15, 23, 42, 0.08)"
_CHART_PAPER_BG = "rgba(0,0,0,0)"


def _restyle_figure_light(fig):
    """Re-themes a Plotly figure (built by backend.analytics) to match the
    light SaaS dashboard palette, without altering any underlying data."""
    try:
        fig.update_layout(
            paper_bgcolor=_CHART_PAPER_BG,
            plot_bgcolor=_CHART_PAPER_BG,
            font=dict(color=_CHART_FONT_COLOR, family="Inter, sans-serif"),
            title_font=dict(color=_CHART_FONT_COLOR, size=15),
            margin=dict(t=50, b=30, l=10, r=10),
        )
        fig.update_xaxes(gridcolor=_CHART_GRID_COLOR, zerolinecolor=_CHART_GRID_COLOR)
        fig.update_yaxes(gridcolor=_CHART_GRID_COLOR, zerolinecolor=_CHART_GRID_COLOR)
    except Exception:
        pass  # purely cosmetic — never break the page if a figure shape is unexpected
    return fig


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def load_css(css_path: str):
    """Injects the custom CSS file into the Streamlit app."""
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def render_sidebar_brand():
    """Renders the sidebar logo, app name, tagline, and version badge."""
    st.sidebar.markdown(
        f"""
        <div class="sidebar-brand">
            <div class="sidebar-brand-logo">🧠</div>
            <div class="sidebar-brand-text">
                <h1>{APP_NAME}</h1>
                <span>{APP_TAGLINE}</span><br/>
                <span class="sidebar-version-tag">{APP_VERSION}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_status(predictor, metrics: dict):
    """Renders a small system-status card in the sidebar."""
    model_ready = predictor is not None and predictor.is_ready()
    dot_class = "dot-success" if model_ready else "dot-danger"
    status_text = "Model Online" if model_ready else "Model Offline"
    accuracy = metrics.get("accuracy", 0) * 100

    st.sidebar.markdown('<div class="sidebar-section-label">System Status</div>', unsafe_allow_html=True)
    st.sidebar.markdown(
        f"""
        <div class="sidebar-status-card">
            <div class="sidebar-status-row">
                <span><span class="sidebar-status-dot {dot_class}"></span>{status_text}</span>
            </div>
            <div class="sidebar-status-row">
                <span>Model Accuracy</span>
                <strong>{accuracy:.1f}%</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_navigation() -> str:
    """
    Renders the full redesigned sidebar navigation list (icons + labels)
    and returns the currently selected page name. Call this once per run
    from app.py in place of the previous inline st.sidebar.radio call.

    Returns:
        str: one of "Home", "Predict Review", "Batch Prediction", "Analytics", "History"
    """
    st.sidebar.markdown('<div class="sidebar-section-label">Navigation</div>', unsafe_allow_html=True)
    page = st.sidebar.radio(
        "Go to",
        [
            "🏠  Home",
            "📝  Predict Review",
            "📂  Batch Prediction",
            "📈  Analytics",
            "🕑  History",
        ],
        label_visibility="collapsed",
    )
    # Strip the leading icon + spacing back to the plain page key app.py expects
    page_key = page.split("  ", 1)[1] if "  " in page else page
    return page_key


def render_sidebar_footer():
    """Renders developer info and a GitHub link in the sidebar footer."""
    st.sidebar.markdown(
        f"""
        <div class="sidebar-footer">
            <div class="sidebar-footer-dev">
                Built &amp; maintained by<br/><strong>{DEVELOPER_NAME}</strong>
            </div>
            <a class="sidebar-github-btn" href="{GITHUB_URL}" target="_blank">
                ⭐ View on GitHub
            </a>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_page_header(eyebrow: str, title: str, subtitle: str):
    """Renders a consistent, professional page header used across all pages."""
    st.markdown(
        f"""
        <div class="page-header">
            <div class="page-header-text">
                <div class="page-eyebrow">{eyebrow}</div>
                <h1>{title}</h1>
                <p>{subtitle}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_kpi_card(
    label: str,
    value,
    col,
    icon: str = "📊",
    icon_variant: str = "primary",
    trend: str = None,
    trend_direction: str = "neutral",
):

    trend_html = ""

    if trend:
        trend_html = f"""
        <span class="kpi-trend kpi-trend-{trend_direction}">
            {trend}
        </span>
        """

    with col:
        st.markdown(
            f"""
            <div class="kpi-card">

                <div class="kpi-icon kpi-icon-{icon_variant}">
                    {icon}
                </div>

                <div class="kpi-value">
                    {value}
                </div>

                <div class="kpi-label">
                    {label}
                </div>

                {trend_html}

            </div>
            """,
            unsafe_allow_html=True,
        )

def render_sentiment_badge(sentiment: str) -> str:
    """Returns an HTML badge string for a sentiment label."""
    if sentiment == "Positive":
        return '<span class="badge badge-success">😊 Positive</span>'
    if sentiment == "Negative":
        return '<span class="badge badge-danger">😞 Negative</span>'
    return '<span class="badge badge-neutral">⚠️ Invalid</span>'


# ---------------------------------------------------------------------------
# Home Page
# ---------------------------------------------------------------------------

def render_home_page(metrics: dict, db_summary: dict):
    render_page_header(
        "🏠 Overview",
        f"Welcome back to {APP_NAME}",
        "Monitor sentiment trends, run predictions, and track model performance — all in one place.",
    )

    # --- Hero section -----------------------------------------------------
    st.markdown(
        f"""
        <div class="hero-section">
            <div class="hero-grid">
                <div class="hero-content">
                    <div class="page-eyebrow">🚀 AI-Powered</div>
                    <div class="hero-title">Understand customer sentiment<br/><span class="accent">in real time.</span></div>
                    <div class="hero-description">
                        This system classifies customer reviews as <strong>Positive</strong> or <strong>Negative</strong>
                        using a TF-IDF + Logistic Regression pipeline, with confidence scoring, batch CSV processing,
                        and a full analytics dashboard — all running locally and privately.
                    </div>
                </div>
                <div class="hero-illustration">🧠</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- KPI grid -----------------------------------------------------------
    st.markdown("### 📊 Model Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.info(f"**Model Accuracy:** {metrics.get('accuracy', 0) * 100:.1f}%")

    with col2:
        st.success(f"**Total Predictions:** {db_summary.get('total', 0)}")

    # --- Quick actions + Recent activity / system status -------------------
    left, right = st.columns([1.4, 1])

    with left:
        st.markdown('<div class="app-card-title">⚡ Quick Actions</div>', unsafe_allow_html=True)
        qa_cols = st.columns(3)
        qa_items = [
            ("📝", "Predict a Review", "Classify a single review instantly"),
            ("📂", "Batch Prediction", "Upload a CSV and process in bulk"),
            ("📈", "View Analytics", "Explore model performance metrics"),
        ]
        for col, (icon, title, desc) in zip(qa_cols, qa_items):
            with col:
                st.markdown(
                    f"""
                    <div class="quick-action-card">
                        <div class="quick-action-icon">{icon}</div>
                        <div class="quick-action-title">{title}</div>
                        <div class="quick-action-desc">{desc}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        st.caption("Use the sidebar navigation to jump to any of these pages.")

        with st.expander("ℹ️ About this system", expanded=False):
            tab1, tab2 = st.tabs(["Tech Stack", "Features"])
            with tab1:
                st.markdown(
                    """
                    - **Frontend:** Streamlit + Custom CSS
                    - **ML Model:** TF-IDF Vectorizer + Logistic Regression
                    - **Database:** SQLite
                    - **Libraries:** pandas, numpy, scikit-learn, nltk, joblib, plotly
                    """
                )
            with tab2:
                st.markdown(
                    """
                    - Single review prediction with confidence score
                    - Batch CSV prediction with downloadable results
                    - Full analytics dashboard (accuracy, precision, recall, F1, confusion matrix)
                    - Persistent prediction history with filtering & export
                    """
                )

    with right:
        st.markdown('<div class="app-card-title">🟢 System Status</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="app-card" style="margin-top: -8px;">
                <div class="status-row"><span>Dataset</span><span class="badge badge-success">Ready</span></div>
                <div class="status-row"><span>Database</span><span class="badge badge-success">Connected</span></div>
                <div class="status-row"><span>Model</span><span class="badge badge-success">Loaded</span></div>
                <div class="status-row"><span>Vectorizer</span><span class="badge badge-success">Loaded</span></div>
                <div class="status-row"><span>Classification</span><span class="badge badge-neutral">Binary (Pos/Neg)</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="app-card-title" style="margin-top: 18px;">🕑 Recent Activity</div>', unsafe_allow_html=True)
        recent_html = _build_recent_activity_html(db_summary)
        st.markdown(f'<div class="app-card" style="margin-top: -8px;">{recent_html}</div>', unsafe_allow_html=True)

    #render_footer()


def _build_recent_activity_html(db_summary: dict) -> str:
    """Builds a tiny activity summary block using only already-available counts."""
    total = db_summary.get("total", 0)
    positive = db_summary.get("positive", 0)
    negative = db_summary.get("negative", 0)

    if total == 0:
        return '<div class="activity-row"><span class="activity-text">No predictions yet — try the Predict Review page.</span></div>'

    rows = [
        ("dot-success", f"{positive} positive predictions logged", "All time"),
        ("dot-danger", f"{negative} negative predictions logged", "All time"),
        ("sidebar-status-dot dot-success", f"{total} total predictions stored in database", "All time"),
    ]
    html = ""
    for dot_class, text, when in rows:
        html += (
            f'<div class="activity-row">'
            f'<span class="activity-dot {dot_class}"></span>'
            f'<span class="activity-text">{text}</span>'
            f'<span class="activity-time">{when}</span>'
            f'</div>'
        )
    return html


# ---------------------------------------------------------------------------
# Predict Review Page (Single Prediction)
# ---------------------------------------------------------------------------

def render_predict_page(predictor, database):
    render_page_header(
        "📝 Single Prediction",
        "Predict Review Sentiment",
        "Enter a customer review below to classify it as Positive or Negative with a confidence score.",
    )

    col_input, col_tips = st.columns([2, 1])

    with col_input:
        with st.container(border=True):
            st.markdown('<div class="app-card-title">✍️ Review Text</div>', unsafe_allow_html=True)
            review_text = st.text_area(
                "Enter a customer review",
                height=150,
                placeholder="e.g. The product quality is amazing and delivery was super fast!",
                label_visibility="collapsed",
            )
            predict_clicked = st.button("🔍 Predict Sentiment", use_container_width=False)

        if predict_clicked:
            with st.spinner("Analyzing review sentiment..."):
                time.sleep(0.15)  # brief, deliberate pause so the spinner is perceptible
                try:
                    result = predictor.predict_single(review_text)
                except PredictorError as exc:
                    st.error(f"⚠️ {exc}")
                    result = None

            if result:
                sentiment = result["sentiment"]
                confidence = result["confidence"]
                css_class = "result-positive" if sentiment == "Positive" else "result-negative"
                icon_class = "result-icon-positive" if sentiment == "Positive" else "result-icon-negative"
                emoji = "😊" if sentiment == "Positive" else "😞"
                sentiment_class = "positive" if sentiment == "Positive" else "negative"

                st.markdown(
                    f"""
                    <div class="result-card {css_class}">
                        <div class="result-icon {icon_class}">{emoji}</div>
                        <div class="result-body">
                            <div class="result-sentiment {sentiment_class}">{sentiment}</div>
                            <div class="result-confidence-label">Confidence Score</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.progress(min(int(confidence), 100), text=f"{confidence:.2f}% confidence")

                try:
                    database.insert_prediction(review_text, sentiment, confidence, source="single")
                    st.toast("Prediction saved to history ✅", icon="✅")
                except DatabaseError as exc:
                    st.warning(f"Prediction succeeded but could not be saved to history: {exc}")

    with col_tips:
        with st.container(border=True):
            st.markdown('<div class="app-card-title">💡 Tips</div>', unsafe_allow_html=True)
            st.markdown(
                """
                <div class="quick-action-desc">
                Write naturally — the model handles full sentences best.<br/><br/>
                Every prediction is automatically saved to your <strong>History</strong> page.<br/><br/>
                Classification is <strong>binary</strong>: Positive or Negative only.
                </div>
                """,
                unsafe_allow_html=True,
            )

    render_footer()


# ---------------------------------------------------------------------------
# Batch Prediction Page
# ---------------------------------------------------------------------------

def render_batch_page(predictor, database):
    render_page_header(
        "📂 Batch Prediction",
        "Bulk Review Classification",
        "Upload a CSV file containing a column named 'review' to classify multiple reviews at once.",
    )

    with st.container(border=True):
        st.markdown('<div class="upload-hint-icon">📤</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Upload CSV file",
            type=["csv"],
            help="The CSV must contain a column named 'review'.",
        )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as exc:
            st.error(f"⚠️ Invalid CSV file: {exc}")
            return

        if "review" not in df.columns:
            st.error("⚠️ The uploaded CSV must contain a column named 'review'.")
            return

        if df.empty:
            st.error("⚠️ The uploaded CSV file is empty.")
            return

        st.success(f"✅ File uploaded successfully — **{len(df)}** reviews found.")
        with st.expander("👀 Preview uploaded data", expanded=True):
            st.dataframe(df.head(5), use_container_width=True)

        run_clicked = st.button("🚀 Run Batch Prediction")

        if run_clicked:
            with st.status("Running batch prediction...", expanded=True) as status:
                st.write("Cleaning and vectorizing reviews...")
                try:
                    results_df = predictor.predict_batch(df["review"].tolist())
                except PredictorError as exc:
                    status.update(label="Batch prediction failed", state="error")
                    st.error(f"⚠️ {exc}")
                    return

                st.write("Classifying sentiment...")
                try:
                    records = list(zip(
                        results_df["review"],
                        results_df["sentiment"],
                        results_df["confidence"],
                        ["batch"] * len(results_df),
                    ))
                    valid_records = [r for r in records if r[1] != "Invalid"]
                    if valid_records:
                        database.insert_batch(valid_records)
                except DatabaseError as exc:
                    st.warning(f"Predictions completed but could not be saved to history: {exc}")

                status.update(label="Batch prediction complete ✅", state="complete")

            st.toast(f"Processed {len(results_df)} reviews", icon="🚀")

            st.markdown('<div class="app-card-title" style="margin-top: 8px;">📋 Results</div>', unsafe_allow_html=True)
            st.dataframe(results_df, use_container_width=True)

            pos_count = int((results_df["sentiment"] == "Positive").sum())
            neg_count = int((results_df["sentiment"] == "Negative").sum())
            invalid_count = int((results_df["sentiment"] == "Invalid").sum())

            cols = st.columns(3)
            render_kpi_card("Positive", pos_count, cols[0], icon="😊", icon_variant="success")
            render_kpi_card("Negative", neg_count, cols[1], icon="😞", icon_variant="danger")
            render_kpi_card("Invalid / Empty", invalid_count, cols[2], icon="⚠️", icon_variant="warning")

            csv_buffer = io.StringIO()
            results_df.to_csv(csv_buffer, index=False)
            st.download_button(
                label="⬇️ Download Results as CSV",
                data=csv_buffer.getvalue(),
                file_name="batch_prediction_results.csv",
                mime="text/csv",
            )

    render_footer()


# ---------------------------------------------------------------------------
# Analytics Page
# ---------------------------------------------------------------------------

def render_analytics_page(metrics: dict, db_summary: dict, history_df: pd.DataFrame):
    render_page_header(
        "📈 Analytics",
        "Model Performance Dashboard",
        "Review classifier performance metrics and sentiment trends across all stored predictions.",
    )

    tab_overview, tab_model, tab_trends = st.tabs(["📊 Overview", "🧪 Model Metrics", "📉 Trends"])

    # --- Overview tab --------------------------------------------------
    with tab_overview:
        summary = an.build_history_summary(history_df)
        cols2 = st.columns(3)

        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            with st.container(border=True):
                st.markdown('<div class="app-card-title">🥧 Sentiment Distribution</div>', unsafe_allow_html=True)
                fig_pie = _restyle_figure_light(an.build_pie_chart(summary["positive"], summary["negative"]))
                st.plotly_chart(fig_pie, use_container_width=True)
        with chart_col2:
            with st.container(border=True):
                st.markdown('<div class="app-card-title">📊 Prediction Counts</div>', unsafe_allow_html=True)
                fig_bar = _restyle_figure_light(an.build_bar_chart(summary["positive"], summary["negative"]))
                st.plotly_chart(fig_bar, use_container_width=True)

    # --- Model metrics tab -----------------------------------------------
    with tab_model:
        st.markdown("###### Classification Metrics")
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Accuracy", f"{metrics.get('accuracy', 0) * 100:.1f}%")
        with m2:
            st.metric("Precision", f"{metrics.get('precision', 0) * 100:.1f}%")
        with m3:
            st.metric("Recall", f"{metrics.get('recall', 0) * 100:.1f}%")
        with m4:
            st.metric("F1 Score", f"{metrics.get('f1_score', 0) * 100:.1f}%")

        st.write("")
        with st.container(border=True):
            st.markdown('<div class="app-card-title">🧩 Confusion Matrix</div>', unsafe_allow_html=True)
            if metrics.get("confusion_matrix"):
                fig_cm = _restyle_figure_light(
                    an.build_confusion_matrix_chart(metrics["confusion_matrix"], metrics.get("labels", ["Positive", "Negative"]))
                )
                st.plotly_chart(fig_cm, use_container_width=True)
            else:
                st.info("Confusion matrix not available. Train the model to generate it.")

        with st.expander("📐 What do these metrics mean?"):
            st.markdown(
                """
                - **Accuracy** — overall percentage of correct predictions.
                - **Precision** — of all reviews predicted Positive, how many actually were.
                - **Recall** — of all actually Positive reviews, how many were correctly found.
                - **F1 Score** — the balance between Precision and Recall.
                """
            )

    # --- Trends tab --------------------------------------------------------
    with tab_trends:
        with st.container(border=True):
            st.markdown('<div class="app-card-title">📉 Prediction Trend Over Time</div>', unsafe_allow_html=True)
            fig_trend = _restyle_figure_light(an.build_history_trend_chart(history_df))
            st.plotly_chart(fig_trend, use_container_width=True)

    render_footer()


# ---------------------------------------------------------------------------
# History Page
# ---------------------------------------------------------------------------

def render_history_page(database):
    render_page_header(
        "🕑 History",
        "Prediction History",
        "Browse, search, filter, sort, and export every prediction ever made by this system.",
    )

    try:
        history_df = database.fetch_all()
    except DatabaseError as exc:
        st.error(f"⚠️ Could not load history: {exc}")
        return

    if history_df.empty:
        st.markdown(
            """
            <div class="app-card" style="text-align:center; padding: 48px 24px;">
                <div style="font-size:42px; margin-bottom:8px;">🗂️</div>
                <div class="app-card-title" style="justify-content:center;">No prediction history yet</div>
                <div class="app-card-subtitle">Run a prediction on the <strong>Predict Review</strong> page to see it appear here.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        render_footer()
        return

    # --- Controls: search, filter, sort -----------------------------------
    with st.container(border=True):
        ctrl_search, ctrl_filter, ctrl_sort, ctrl_clear = st.columns([2, 1, 1, 1])

        with ctrl_search:
            search_term = st.text_input("🔎 Search reviews", placeholder="Search by keyword...")
        with ctrl_filter:
            sentiment_filter = st.selectbox("Filter", ["All", "Positive", "Negative"])
        with ctrl_sort:
            sort_option = st.selectbox("Sort by", ["Newest First", "Oldest First", "Confidence (High→Low)", "Confidence (Low→High)"])
        with ctrl_clear:
            st.write("")
            clear_clicked = st.button("🗑️ Clear All", use_container_width=True)

    if clear_clicked:
        try:
            database.clear_history()
            st.success("History cleared successfully.")
            st.rerun()
        except DatabaseError as exc:
            st.error(f"⚠️ Failed to clear history: {exc}")

    # --- Apply search / filter / sort (frontend-only data shaping) --------
    display_df = history_df.copy()

    if sentiment_filter != "All":
        display_df = display_df[display_df["prediction"] == sentiment_filter]

    if search_term:
        display_df = display_df[display_df["review"].str.contains(search_term, case=False, na=False)]

    if sort_option == "Newest First":
        display_df = display_df.sort_values("id", ascending=False)
    elif sort_option == "Oldest First":
        display_df = display_df.sort_values("id", ascending=True)
    elif sort_option == "Confidence (High→Low)":
        display_df = display_df.sort_values("confidence", ascending=False)
    elif sort_option == "Confidence (Low→High)":
        display_df = display_df.sort_values("confidence", ascending=True)
    st.markdown("### 📊 Summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Matching Records", len(history_df))

    with col2:
        st.metric("Positive", len(history_df[history_df["prediction"] == "Positive"]))

    with col3:
        st.metric("Negative", len(history_df[history_df["prediction"] == "Negative"]))
  

    # --- Pagination ---------------------------------------------------------
    page_size = 15
    total_rows = len(display_df)
    total_pages = max(1, (total_rows - 1) // page_size + 1)

    if "history_page_num" not in st.session_state:
        st.session_state.history_page_num = 1
    st.session_state.history_page_num = min(st.session_state.history_page_num, total_pages)

    with st.container(border=True):
        st.markdown('<div class="app-card-title">📋 Prediction Records</div>', unsafe_allow_html=True)

        start_idx = (st.session_state.history_page_num - 1) * page_size
        end_idx = start_idx + page_size
        page_df = display_df.iloc[start_idx:end_idx]

        st.dataframe(page_df, use_container_width=True, height=420)

        pag_prev, pag_label, pag_next = st.columns([1, 2, 1])
        with pag_prev:
            if st.button("⬅️ Previous", disabled=st.session_state.history_page_num <= 1, use_container_width=True):
                st.session_state.history_page_num -= 1
                st.rerun()
        with pag_label:
            st.markdown(
                f"<div style='text-align:center; color:#64748B; padding-top:8px;'>"
                f"Page {st.session_state.history_page_num} of {total_pages} &nbsp;·&nbsp; {total_rows} records"
                f"</div>",
                unsafe_allow_html=True,
            )
        with pag_next:
            if st.button("Next ➡️", disabled=st.session_state.history_page_num >= total_pages, use_container_width=True):
                st.session_state.history_page_num += 1
                st.rerun()

    csv_buffer = io.StringIO()
    display_df.to_csv(csv_buffer, index=False)
    st.download_button(
        label="⬇️ Export Filtered Results as CSV",
        data=csv_buffer.getvalue(),
        file_name="prediction_history.csv",
        mime="text/csv",
    )

    render_footer()
def render_footer():
    import streamlit as st

    st.markdown(
        """
        <div style="
            text-align:center;
            margin-top:40px;
            padding:20px;
            color:#6B7280;
            font-size:14px;
            border-top:1px solid #E5E7EB;
        ">
            © 2026 AI-Powered Sentiment Analysis System<br>
            Built by <b>Madhan Raj P</b>
        </div>
        """,
        unsafe_allow_html=True,
    )