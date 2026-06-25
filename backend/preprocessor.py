"""
preprocessor.py
Text cleaning and preprocessing utilities for the sentiment analysis pipeline.
Uses NLTK for stopword removal and lemmatization.
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def _ensure_nltk_resources():
    """Downloads required NLTK resources quietly if not already present."""
    resources = [
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
    ]
    for path, package in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(package, quiet=True)


_ensure_nltk_resources()

_STOPWORDS = set(stopwords.words("english"))
# Keep negation words - they carry strong sentiment signal and removing them
# would flip the meaning of phrases like "not good" into "good".
_NEGATIONS = {
    "no", "not", "nor", "never", "none", "n't", "cannot", "without",
    "isn't", "wasn't", "aren't", "weren't", "don't", "doesn't", "didn't",
    "won't", "wouldn't", "couldn't", "shouldn't", "can't",
}
_STOPWORDS = _STOPWORDS - _NEGATIONS

_LEMMATIZER = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """
    Cleans and normalizes a single piece of review text.

    Steps:
        1. Lowercase
        2. Remove URLs, HTML tags, and non-alphabetic characters
        3. Tokenize on whitespace
        4. Remove stopwords (excluding negations)
        5. Lemmatize remaining tokens

    Args:
        text: Raw review text.

    Returns:
        Cleaned, lemmatized text as a single string.
    """
    if text is None:
        return ""

    text = str(text)
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = text.split()
    tokens = [t for t in tokens if t not in _STOPWORDS and len(t) > 1]
    tokens = [_LEMMATIZER.lemmatize(t) for t in tokens]

    return " ".join(tokens)


def clean_batch(texts) -> list:
    """
    Cleans a list/Series of review texts.

    Args:
        texts: Iterable of raw review strings.

    Returns:
        List of cleaned strings.
    """
    return [clean_text(t) for t in texts]


def validate_review_text(text: str) -> tuple:
    """
    Validates that a review string is usable for prediction.

    Args:
        text: Raw input text from the user.

    Returns:
        (is_valid, message) tuple. is_valid is False if the review is
        empty, whitespace-only, or too short to be meaningful.
    """
    if text is None:
        return False, "Review text is empty. Please enter a review."

    stripped = text.strip()
    if len(stripped) == 0:
        return False, "Review text is empty. Please enter a review."

    if len(stripped) < 3:
        return False, "Review text is too short. Please enter a more detailed review."

    cleaned = clean_text(stripped)
    if len(cleaned) == 0:
        return False, "Review text contains no meaningful words after cleaning. Please rephrase."

    return True, ""
