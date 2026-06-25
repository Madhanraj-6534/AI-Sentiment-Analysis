# 🧠 AI-Powered Customer Review Sentiment Analysis System

A complete, production-ready **binary sentiment analysis** web application built with **Streamlit**, **scikit-learn (TF-IDF + Logistic Regression)**, and **SQLite**. Classifies customer reviews as **Positive** or **Negative**, with confidence scores, batch CSV prediction, a full analytics dashboard, and persistent prediction history.

> ⚠️ This project intentionally uses **binary classification only** (Positive / Negative) — no Neutral class — for improved accuracy and reliability.

---

## ✨ Features

- **Single Review Prediction** — enter a review and get an instant Positive/Negative classification with a confidence score.
- **Batch Prediction** — upload a CSV of reviews, classify them all, and download the results.
- **Analytics Dashboard** — Accuracy, Precision, Recall, F1 Score, Confusion Matrix, and sentiment distribution charts.
- **Prediction History** — every prediction is logged to a local SQLite database; filter, view, export, or clear history.
- **Automatic Dataset Generation** — a realistic synthetic dataset of 500+ positive and 500+ negative reviews is generated locally on first run (no internet download required).
- **Automatic Model Training** — the TF-IDF vectorizer and Logistic Regression model train automatically on first launch if no saved model is found, then load instantly on every subsequent run.
- **Professional Dark Blue UI** — custom CSS theme, KPI cards, sidebar navigation, recruiter-friendly layout.
- **Robust Error Handling** — empty reviews, invalid CSVs, missing models, and database errors are all handled gracefully.

---

## 🛠️ Tech Stack

| Layer          | Technology                              |
|----------------|------------------------------------------|
| Frontend       | Streamlit + Custom CSS                  |
| Backend        | Python                                   |
| ML Model       | TF-IDF Vectorizer + Logistic Regression |
| Database       | SQLite                                  |
| Visualization  | Plotly                                  |
| NLP            | NLTK (stopwords + lemmatization)        |
| Persistence    | Joblib                                  |

---

## 📁 Folder Structure

```
Sentiment_Analysis/
│
├── app.py                     # Main Streamlit entry point + startup checks
├── requirements.txt           # Python dependencies
├── README.md                  # This file
│
├── backend/
│   ├── __init__.py
│   ├── dataset_generator.py   # Generates the synthetic review dataset
│   ├── preprocessor.py        # Text cleaning, stopword removal, lemmatization
│   ├── trainer.py             # Trains TF-IDF + Logistic Regression model
│   ├── predictor.py           # Loads model, runs single/batch predictions
│   ├── database.py            # SQLite storage layer
│   └── analytics.py           # Computes metrics & builds Plotly charts
│
├── frontend/
│   ├── __init__.py
│   ├── ui.py                  # Streamlit page rendering functions
│   └── styles.css             # Dark blue custom theme
│
├── data/
│   └── reviews.csv            # Auto-generated on first run
│
├── models/
│   ├── sentiment_model.pkl    # Auto-generated on first run
│   └── vectorizer.pkl         # Auto-generated on first run
│
├── database/
│   └── sentiment.db           # Auto-generated on first run
│
├── assets/                    # Reserved for images/icons
└── static/                    # Reserved for static files
```

---

## ⚙️ Installation

**Requirements:** Python 3.10+

1. Clone or download this project folder.
2. Open the folder in **VS Code** (or any terminal).
3. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS / Linux
   source venv/bin/activate
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## ▶️ Run Instructions

From the project root folder, run:

```bash
python -m streamlit run app.py
```

Then open the URL shown in the terminal (typically `http://localhost:8501`) in your browser.

### What happens on first run

The app performs automatic startup checks, in order:

1. **Check dataset** — generates `data/reviews.csv` (520 positive + 520 negative reviews) if missing.
2. **Check database** — creates `database/sentiment.db` and the `predictions` table if missing.
3. **Check model** — looks for `models/sentiment_model.pkl` and `models/vectorizer.pkl`.
4. **Train model** — if model files are missing, automatically trains a TF-IDF + Logistic Regression pipeline on the dataset and saves it via Joblib.
5. **Load model** — loads the trained model into memory, ready for predictions.

Subsequent runs skip straight to loading the saved model — no retraining needed.

---

## 🖥️ Pages

| Page              | Description                                                              |
|-------------------|---------------------------------------------------------------------------|
| **Home**          | Overview KPIs and project summary                                        |
| **Predict Review**| Single review input with sentiment + confidence output                   |
| **Batch Prediction**| Upload a CSV (`review` column) and download predictions               |
| **Analytics**     | Accuracy, Precision, Recall, F1, Confusion Matrix, sentiment charts      |
| **History**       | Filterable, exportable log of every prediction ever made                 |

---

## 📊 Screenshots

> _Add your own screenshots here after running the app locally._

- `assets/home_page.png`
- `assets/predict_page.png`
- `assets/analytics_page.png`
- `assets/history_page.png`

---

## 🧪 Model Details

- **Vectorizer:** `TfidfVectorizer` (unigrams + bigrams, max 5000 features)
- **Classifier:** `LogisticRegression` (balanced class weights, max_iter=1000)
- **Train/Test Split:** 80/20, stratified
- **Metrics tracked:** Accuracy, Precision, Recall, F1 Score, Confusion Matrix

---

## ❗ Error Handling Coverage

- Empty or whitespace-only review text
- CSV files missing the required `review` column
- Empty or corrupted CSV uploads
- Missing model/vectorizer files (auto-retrains)
- Missing dataset (auto-regenerates)
- SQLite connection/transaction failures

---

## 📄 License

This project is provided as-is for educational and portfolio purposes.
