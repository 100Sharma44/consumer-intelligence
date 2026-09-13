"""Feature generation and model inference for purchase intent."""

from datetime import datetime
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from app.db.connection import get_db_connection


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "purchase_intent_tuned.joblib"


class UserNotFoundError(Exception):
    """Raised when inference is requested for a user that does not exist."""


@lru_cache(maxsize=1)
def load_model_artifact():
    """Load the trained pipeline once, including its feature contract and threshold."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Tuned model artifact not found: {MODEL_PATH}")

    artifact = joblib.load(MODEL_PATH)
    required_keys = {"pipeline", "feature_columns", "classification_threshold"}
    missing_keys = required_keys.difference(artifact)
    if missing_keys:
        raise ValueError(f"Model artifact is missing required keys: {sorted(missing_keys)}")
    return artifact


def load_user_events(user_id, prediction_timestamp):
    """Load one user's behavior available at the prediction timestamp only."""
    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1 FROM users WHERE user_id = %s", (user_id,))
            if cursor.fetchone() is None:
                raise UserNotFoundError(f"User {user_id} does not exist.")

            cursor.execute(
                """
                SELECT e.product_id, e.event_type, e.event_time, p.category,
                       p.price::double precision AS price
                FROM events AS e
                LEFT JOIN products AS p ON p.product_id = e.product_id
                WHERE e.user_id = %s
                  AND e.event_time IS NOT NULL
                  AND e.event_time <= %s
                ORDER BY e.event_time, e.event_id
                """,
                (user_id, prediction_timestamp),
            )
            rows = cursor.fetchall()

    return pd.DataFrame(rows, columns=["product_id", "event_type", "event_time", "category", "price"])


def build_feature_row(events, prediction_timestamp, feature_columns):
    """Mirror the training feature definitions using only historical event rows."""
    features = {column: 0.0 for column in feature_columns}
    if events.empty:
        return pd.DataFrame([features], columns=feature_columns)

    events = events.copy()
    events["event_time"] = pd.to_datetime(events["event_time"])
    event_counts = events["event_type"].value_counts()
    for event_type in ("search", "view", "wishlist", "cart"):
        features[f"{event_type}_count"] = int(event_counts.get(event_type, 0))

    features["purchase_count_before_prediction"] = int(event_counts.get("purchase", 0))
    features["total_event_count"] = int(len(events))
    features["activity_days"] = int(events["event_time"].dt.normalize().nunique())
    features["recency_hours"] = (
        prediction_timestamp - events["event_time"].max()
    ).total_seconds() / 3600
    features["distinct_products_viewed"] = int(
        events.loc[events["event_type"] == "view", "product_id"].nunique()
    )
    features["distinct_categories_interacted"] = int(events["category"].nunique())
    features["average_interacted_product_price"] = float(events["price"].mean())

    cart_prices = events.loc[events["event_type"] == "cart", "price"]
    features["average_carted_product_price"] = float(cart_prices.mean()) if not cart_prices.empty else 0.0
    return pd.DataFrame([features], columns=feature_columns)


def intent_level(probability, classification_threshold):
    """Convert a probability to a presentation-level label without changing the classifier threshold."""
    if probability < classification_threshold:
        return "low"

    high_intent_cutoff = classification_threshold + (1 - classification_threshold) / 2
    if probability < high_intent_cutoff:
        return "medium"
    return "high"


def predict_purchase_intent(user_id):
    """Generate historical features and return a single purchase-intent prediction."""
    artifact = load_model_artifact()
    prediction_timestamp = datetime.now()
    events = load_user_events(user_id, prediction_timestamp)
    feature_row = build_feature_row(events, prediction_timestamp, artifact["feature_columns"])

    probability = float(artifact["pipeline"].predict_proba(feature_row)[0, 1])
    threshold = float(artifact["classification_threshold"])
    predicted_intent = int(probability >= threshold)
    return {
        "user_id": user_id,
        "purchase_intent_probability": probability,
        "predicted_purchase_intent": predicted_intent,
        "intent_level": intent_level(probability, threshold),
    }
