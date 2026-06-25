"""
app.py
Main Streamlit entry point for the AI-Powered Customer Review
Sentiment Analysis System.

Run with:
    python -m streamlit run app.py
"""

import os
import sys

import streamlit as st

# Ensure the project root is on the path so 'backend' and 'frontend' packages
# resolve correctly regardless of the working directory Streamlit is launched from.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.dataset_generator import ensure_dataset
from backend.trainer import train_if_needed, models_exist
from backend.predictor import SentimentPredictor, PredictorError
from backend.database import Database, DatabaseError
from frontend import ui


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "reviews.csv")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "sentiment_model.pkl")
VECTORIZER_PATH = os.path.join(PROJECT_ROOT, "models", "vectorizer.pkl")
DB_PATH = os.path.join(PROJECT_ROOT, "database", "sentiment.db")
CSS_PATH = os.path.join(PROJECT_ROOT, "frontend", "styles.css")


# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AI Sentiment Analysis System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Startup checks — cached so they only run once per session/process.
# Order: dataset -> database -> model -> train if missing -> load model.
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def initialize_system():
    """
    Performs all startup checks and returns initialized objects.

    Returns:
        dict with keys: predictor, database, metrics, init_error
    """
    init_error = None
    metrics = {}
    predictor = None
    database = None

    try:
        # 1. Check / generate dataset
        ensure_dataset(DATA_PATH)

        # 2. Check / initialize database
        database = Database(DB_PATH)

        # 3 & 4. Check model, train if missing
        metrics = train_if_needed(DATA_PATH, MODEL_PATH, VECTORIZER_PATH)

        # 5. Load model automatically
        if models_exist(MODEL_PATH, VECTORIZER_PATH):
            predictor = SentimentPredictor(MODEL_PATH, VECTORIZER_PATH).load()
        else:
            init_error = "Model files could not be created. Please check write permissions for the 'models' folder."

    except (PredictorError, DatabaseError) as exc:
        init_error = str(exc)
    except Exception as exc:  # noqa: BLE001 - surface any startup failure to the UI
        init_error = f"Unexpected startup error: {exc}"

    return {
        "predictor": predictor,
        "database": database,
        "metrics": metrics,
        "init_error": init_error,
    }


def main():
    from frontend.ui import load_css

    load_css("frontend/styles.css")

    system = initialize_system()

    if system["init_error"]:
        st.error(f"🚨 Application failed to initialize: {system['init_error']}")
        st.info("Try deleting the 'models' and 'database' folders and restarting the app to force a clean rebuild.")
        st.stop()

    predictor = system["predictor"]
    database = system["database"]
    metrics = system["metrics"]

    # Sidebar — brand, navigation, system status, and footer (all rendering
    # logic lives in frontend/ui.py; this just calls it).
    ui.render_sidebar_brand()
    page = ui.render_sidebar_navigation()
    ui.render_sidebar_status(predictor, metrics)
    ui.render_sidebar_footer()

    # Fetch shared data needed across pages
    try:
        db_summary = database.fetch_summary()
    except DatabaseError as exc:
        st.warning(f"Could not load database summary: {exc}")
        db_summary = {"total": 0, "positive": 0, "negative": 0}

    try:
        history_df = database.fetch_all()
    except DatabaseError:
        history_df = None

    # Route to the selected page
    if page == "Home":
        ui.render_home_page(metrics, db_summary)
    elif page == "Predict Review":
        ui.render_predict_page(predictor, database)
    elif page == "Batch Prediction":
        ui.render_batch_page(predictor, database)
    elif page == "Analytics":
        ui.render_analytics_page(metrics, db_summary, history_df)
    elif page == "History":
        ui.render_history_page(database)


if __name__ == "__main__":
    main()
