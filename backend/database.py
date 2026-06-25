"""
database.py
SQLite database layer for storing prediction history.
"""

import os
import sqlite3
from datetime import datetime
from contextlib import contextmanager
import pandas as pd


class DatabaseError(Exception):
    """Raised when a database operation fails."""
    pass


class Database:
    """Handles all SQLite operations for the sentiment analysis app."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._initialize()

    @contextmanager
    def _get_connection(self):
        conn = None
        try:
            conn = sqlite3.connect(self.db_path, timeout=10)
            conn.row_factory = sqlite3.Row
            yield conn
            conn.commit()
        except sqlite3.Error as exc:
            if conn:
                conn.rollback()
            raise DatabaseError(f"Database error: {exc}") from exc
        finally:
            if conn:
                conn.close()

    def _initialize(self):
        """Creates the predictions table if it does not already exist."""
        try:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS predictions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        review TEXT NOT NULL,
                        prediction TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        source TEXT DEFAULT 'single',
                        timestamp TEXT NOT NULL
                    )
                    """
                )
        except DatabaseError as exc:
            raise DatabaseError(f"Failed to initialize database: {exc}") from exc

    def insert_prediction(self, review: str, prediction: str, confidence: float, source: str = "single"):
        """Inserts a single prediction record into the database."""
        try:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO predictions (review, prediction, confidence, source, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (review, prediction, float(confidence), source, datetime.now().isoformat(timespec="seconds")),
                )
        except DatabaseError as exc:
            raise DatabaseError(f"Failed to insert prediction: {exc}") from exc

    def insert_batch(self, records: list):
        """
        Inserts multiple prediction records at once.

        Args:
            records: List of tuples (review, prediction, confidence, source)
        """
        try:
            timestamp = datetime.now().isoformat(timespec="seconds")
            rows = [(r[0], r[1], float(r[2]), r[3] if len(r) > 3 else "batch", timestamp) for r in records]
            with self._get_connection() as conn:
                conn.executemany(
                    """
                    INSERT INTO predictions (review, prediction, confidence, source, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    rows,
                )
        except DatabaseError as exc:
            raise DatabaseError(f"Failed to insert batch predictions: {exc}") from exc

    def fetch_all(self) -> pd.DataFrame:
        """Returns all prediction history as a DataFrame, most recent first."""
        try:
            with self._get_connection() as conn:
                df = pd.read_sql_query(
                    "SELECT id, review, prediction, confidence, source, timestamp "
                    "FROM predictions ORDER BY id DESC",
                    conn,
                )
            return df
        except DatabaseError as exc:
            raise DatabaseError(f"Failed to fetch history: {exc}") from exc

    def fetch_summary(self) -> dict:
        """Returns aggregate counts used by the analytics dashboard."""
        try:
            with self._get_connection() as conn:
                total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
                positive = conn.execute(
                    "SELECT COUNT(*) FROM predictions WHERE prediction = 'Positive'"
                ).fetchone()[0]
                negative = conn.execute(
                    "SELECT COUNT(*) FROM predictions WHERE prediction = 'Negative'"
                ).fetchone()[0]
            return {"total": total, "positive": positive, "negative": negative}
        except DatabaseError as exc:
            raise DatabaseError(f"Failed to fetch summary: {exc}") from exc

    def clear_history(self):
        """Deletes all prediction history records."""
        try:
            with self._get_connection() as conn:
                conn.execute("DELETE FROM predictions")
        except DatabaseError as exc:
            raise DatabaseError(f"Failed to clear history: {exc}") from exc
