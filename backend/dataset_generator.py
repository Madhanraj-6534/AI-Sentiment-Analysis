"""
dataset_generator.py
Generates a realistic synthetic customer review dataset locally
(no internet download). Produces at least 500 positive and 500
negative reviews and writes them to data/reviews.csv.
"""

import os
import random
import pandas as pd

random.seed(42)

# ---------------------------------------------------------------------------
# Building blocks used to compose realistic, varied review sentences.
# Combining subjects + phrases produces hundreds of unique, natural reviews.
# ---------------------------------------------------------------------------

PRODUCTS = [
    "product", "item", "order", "purchase", "device", "gadget", "phone case",
    "headphones", "shoes", "jacket", "laptop", "watch", "blender", "backpack",
    "camera", "speaker", "charger", "mattress", "chair", "monitor", "keyboard",
    "mouse", "tablet", "vacuum cleaner", "coffee maker", "air fryer", "router",
    "television", "sofa", "table lamp", "bookshelf", "winter coat", "sneakers",
    "sunglasses", "wallet", "handbag", "perfume", "skincare cream", "toy",
    "board game", "fitness tracker", "bluetooth earbuds", "printer", "drone",
    "microwave", "refrigerator", "washing machine", "dining set", "curtains",
    "kids' bicycle", "running shorts"
]

POSITIVE_OPENERS = [
    "I absolutely love this {p}.",
    "This {p} exceeded my expectations.",
    "Fantastic {p}, totally worth the price.",
    "I am extremely happy with this {p}.",
    "What a wonderful {p}, highly recommend it.",
    "This {p} works perfectly and looks great.",
    "Amazing quality for this {p}.",
    "Best {p} I have purchased this year.",
    "I'm impressed with how well this {p} performs.",
    "This {p} is exactly what I was looking for.",
    "Outstanding {p}, exceeded all my expectations.",
    "I couldn't be happier with this {p}.",
    "Superb {p}, the quality is excellent.",
    "This {p} is a fantastic addition to my home.",
    "Really happy with this {p}, great value for money.",
]

POSITIVE_MIDDLES = [
    "The build quality is excellent and it feels premium.",
    "Delivery was fast and the packaging was secure.",
    "Customer service was helpful and responded quickly.",
    "It works exactly as described in the listing.",
    "The design is sleek and very comfortable to use.",
    "It arrived earlier than expected and in perfect condition.",
    "The price point makes it an absolute bargain.",
    "Setup was quick and the instructions were clear.",
    "It performs better than similar products I've tried before.",
    "The material feels durable and well made.",
    "Battery life is impressive and lasts all day.",
    "The colors are vibrant and exactly as shown in pictures.",
    "It fits perfectly and is very easy to use.",
    "I have used it daily and it still works flawlessly.",
    "The features included are genuinely useful and well thought out.",
]

POSITIVE_CLOSERS = [
    "I will definitely buy from this brand again.",
    "Highly recommend it to anyone considering this purchase.",
    "Five stars, no complaints at all.",
    "Worth every penny and more.",
    "I am extremely satisfied with this purchase.",
    "Will be ordering more as gifts for my family.",
    "This has become one of my favorite purchases.",
    "I would rate this a perfect ten out of ten.",
    "Definitely exceeded my expectations for the price.",
    "I'm already planning to order another one.",
    "Couldn't ask for a better product.",
    "This is going straight onto my list of favorites.",
]

NEGATIVE_OPENERS = [
    "I am very disappointed with this {p}.",
    "This {p} broke after just a few days.",
    "Terrible quality for this {p}, would not recommend.",
    "I regret buying this {p}.",
    "What a waste of money this {p} turned out to be.",
    "This {p} did not work as advertised.",
    "Poor quality {p}, not worth the price.",
    "Worst {p} I have ever purchased.",
    "I'm frustrated with how this {p} performed.",
    "This {p} is far from what was described in the listing.",
    "Awful experience with this {p}.",
    "I'm extremely unhappy with this {p}.",
    "Disappointing {p}, the quality is really bad.",
    "This {p} stopped working within a week.",
    "Really unhappy with this {p}, total waste of money.",
]

NEGATIVE_MIDDLES = [
    "The build quality feels cheap and flimsy.",
    "Delivery was delayed and the packaging was damaged.",
    "Customer service was unhelpful and slow to respond.",
    "It does not work as described in the listing.",
    "The design is uncomfortable and poorly thought out.",
    "It arrived late and in damaged condition.",
    "The price is far too high for what you actually get.",
    "Setup was confusing and the instructions were unclear.",
    "It performs worse than similar products I've tried before.",
    "The material feels cheap and breaks easily.",
    "Battery life is terrible and drains within hours.",
    "The colors look faded compared to the pictures shown.",
    "It does not fit properly and is hard to use.",
    "After a few uses it stopped working completely.",
    "The features promised are missing or simply don't work.",
]

NEGATIVE_CLOSERS = [
    "I will never buy from this brand again.",
    "I would not recommend this to anyone.",
    "One star, full of complaints.",
    "Not worth the money at all.",
    "I am extremely dissatisfied with this purchase.",
    "I am returning this immediately for a refund.",
    "This has become one of my worst purchases.",
    "I would rate this a one out of ten.",
    "Completely failed to meet my expectations.",
    "I already regret ordering this.",
    "Could not be more disappointed with this product.",
    "This is going straight back to the seller.",
]


def _compose(openers, middles, closers, product):
    """Combine random sentence fragments into a coherent review."""
    opener = random.choice(openers).format(p=product)
    middle = random.choice(middles)
    closer = random.choice(closers)
    return f"{opener} {middle} {closer}"


def generate_dataset(n_positive: int = 520, n_negative: int = 520) -> pd.DataFrame:
    """
    Generates a synthetic dataset of customer reviews.

    Returns:
        pd.DataFrame with columns ['review', 'sentiment']
        sentiment is either 'Positive' or 'Negative'.
    """
    rows = []
    seen = set()

    # Positive reviews
    attempts = 0
    while len([r for r in rows if r[1] == "Positive"]) < n_positive and attempts < n_positive * 20:
        attempts += 1
        product = random.choice(PRODUCTS)
        review = _compose(POSITIVE_OPENERS, POSITIVE_MIDDLES, POSITIVE_CLOSERS, product)
        if review not in seen:
            seen.add(review)
            rows.append((review, "Positive"))

    # Negative reviews
    attempts = 0
    while len([r for r in rows if r[1] == "Negative"]) < n_negative and attempts < n_negative * 20:
        attempts += 1
        product = random.choice(PRODUCTS)
        review = _compose(NEGATIVE_OPENERS, NEGATIVE_MIDDLES, NEGATIVE_CLOSERS, product)
        if review not in seen:
            seen.add(review)
            rows.append((review, "Negative"))

    random.shuffle(rows)
    df = pd.DataFrame(rows, columns=["review", "sentiment"])
    return df


def ensure_dataset(csv_path: str) -> pd.DataFrame:
    """
    Ensures a dataset CSV exists at csv_path. If missing, generates one.
    Returns the loaded DataFrame.
    """
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            if {"review", "sentiment"}.issubset(df.columns) and len(df) >= 100:
                return df
        except Exception:
            pass  # fall through to regeneration if file is corrupt

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df = generate_dataset()
    df.to_csv(csv_path, index=False)
    return df


if __name__ == "__main__":
    out_path = os.path.join(os.path.dirname(__file__), "..", "data", "reviews.csv")
    out_path = os.path.normpath(out_path)
    dataframe = ensure_dataset(out_path)
    print(f"Dataset generated with {len(dataframe)} rows at {out_path}")
    print(dataframe["sentiment"].value_counts())
