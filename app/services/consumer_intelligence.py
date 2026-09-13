"""Transparent behavioral consumer-intelligence signals built on existing data.

Behavioral price sensitivity describes observed product-price interaction patterns.
It is not an estimate of income, wealth, affordability, or financial status.
"""

from datetime import datetime
from math import exp

import pandas as pd

from app.db.connection import get_db_connection
from app.services import purchase_intent


# All scores are deterministic and based exclusively on observed behavior.
INTEREST_EVENT_WEIGHTS = {"search": 1, "view": 2, "wishlist": 4, "cart": 6, "purchase": 3}
CURRENT_INTENT_EVENT_WEIGHTS = {"search": 1, "view": 2, "wishlist": 5, "cart": 8}
INTEREST_RECENCY_DECAY_DAYS = 30
CURRENT_INTENT_RECENCY_DECAY_DAYS = 14
CURRENT_INTENT_SCORE_SCALE = 12
PURCHASE_PROPENSITY_THRESHOLD = 0.6179


def _empty_events():
    return pd.DataFrame(columns=["product_id", "event_type", "event_time", "category", "price"])


def _prepare_events(events):
    """Return a copy with valid timestamps, suitable for deterministic calculations."""
    if events.empty:
        return _empty_events()
    prepared = events.copy()
    prepared["event_time"] = pd.to_datetime(prepared["event_time"])
    return prepared.dropna(subset=["event_time"])


def calculate_primary_interest(events, as_of):
    """Find the category with the largest recency-weighted engagement share.

    Each event contributes event_weight * exp(-age_in_days / 30). The reported
    score is the leading category's share of all category engagement, from 0 to 100.
    """
    events = _prepare_events(events).dropna(subset=["category"])
    if events.empty:
        return {"primary_interest": None, "interest_score": 0.0}

    weighted_scores = {}
    for event in events.itertuples(index=False):
        age_days = max(0.0, (as_of - event.event_time).total_seconds() / 86400)
        weight = INTEREST_EVENT_WEIGHTS.get(event.event_type, 0)
        contribution = weight * exp(-age_days / INTEREST_RECENCY_DECAY_DAYS)
        weighted_scores[event.category] = weighted_scores.get(event.category, 0.0) + contribution

    total_score = sum(weighted_scores.values())
    if total_score == 0:
        return {"primary_interest": None, "interest_score": 0.0}

    primary_category, primary_score = max(weighted_scores.items(), key=lambda item: item[1])
    return {
        "primary_interest": primary_category,
        "interest_score": round(100 * primary_score / total_score, 2),
    }


def calculate_current_intent(events, as_of):
    """Score near-term shopping intent from recency-weighted funnel actions.

    The raw score is sum(event_weight * exp(-age_in_days / 14)). It is converted
    to 0-100 with 100 * (1 - exp(-raw_score / 12)). Scores below 30 are low,
    30-59.99 are medium, and scores of 60 or greater are high.
    """
    events = _prepare_events(events)
    raw_score = 0.0
    for event in events.itertuples(index=False):
        weight = CURRENT_INTENT_EVENT_WEIGHTS.get(event.event_type, 0)
        if weight:
            age_days = max(0.0, (as_of - event.event_time).total_seconds() / 86400)
            raw_score += weight * exp(-age_days / CURRENT_INTENT_RECENCY_DECAY_DAYS)

    intent_score = 100 * (1 - exp(-raw_score / CURRENT_INTENT_SCORE_SCALE))
    if intent_score >= 60:
        intent_level = "high"
    elif intent_score >= 30:
        intent_level = "medium"
    else:
        intent_level = "low"
    return {"intent_score": round(intent_score, 2), "intent_level": intent_level}


def calculate_behavioral_price_sensitivity(events, catalog_prices):
    """Estimate behavioral price sensitivity from observed price interactions only.

    The reference price is the average interacted price, except when cart events
    exist: it becomes 60% average carted price plus 40% average interacted price.
    Sensitivity = clamp(50 + 50 * (1 - reference_price / catalog_median), 0, 100).
    Lower observed price levels relative to the catalog median therefore yield a
    higher behavioral price-sensitivity score. This is strictly a description of
    observed shopping behavior, not income, wealth, affordability, or financial status.
    """
    events = _prepare_events(events)
    available_prices = pd.Series(catalog_prices, dtype="float64").dropna()
    interacted_prices = pd.to_numeric(events.get("price", pd.Series(dtype="float64")), errors="coerce").dropna()
    if available_prices.empty or interacted_prices.empty:
        return {"price_sensitivity_score": 0.0, "price_sensitivity_level": "unknown"}

    catalog_median = float(available_prices.median())
    if catalog_median <= 0:
        return {"price_sensitivity_score": 0.0, "price_sensitivity_level": "unknown"}
    interacted_average = float(interacted_prices.mean())
    cart_prices = pd.to_numeric(
        events.loc[events["event_type"] == "cart", "price"], errors="coerce"
    ).dropna()
    reference_price = (
        0.6 * float(cart_prices.mean()) + 0.4 * interacted_average
        if not cart_prices.empty
        else interacted_average
    )
    score = max(0.0, min(100.0, 50 + 50 * (1 - reference_price / catalog_median)))
    if score >= 65:
        level = "high"
    elif score >= 35:
        level = "medium"
    else:
        level = "low"
    return {"price_sensitivity_score": round(score, 2), "price_sensitivity_level": level}


def assign_consumer_segment(intent_level, price_sensitivity_level, purchase_probability, threshold):
    """Assign a segment from current behavior and ML purchase propensity.

    ``purchase_probability`` is the ML-generated seven-day purchase propensity.
    ``intent_level`` is the separate rule-based, current behavioral shopping intent.
    """
    if intent_level == "high" and purchase_probability >= threshold:
        if price_sensitivity_level == "high":
            return "high_intent_value_seeker"
        return "high_intent_premium_shopper"
    if intent_level == "low" and purchase_probability >= threshold:
        return "latent_high_propensity"
    if intent_level in {"medium", "high"}:
        return "active_researcher"
    return "low_intent_browser"


def make_advertising_decision(consumer_intelligence_output):
    """Return an explainable V1 advertising decision from existing intelligence signals.

    The saved ML purchase-propensity threshold is 0.6179. No behavioral score is
    recalculated here: this function only maps already-computed intelligence to a
    recommended objective, category, messaging style, and concise reason.
    """
    intent_level = consumer_intelligence_output["intent_level"]
    purchase_probability = float(consumer_intelligence_output["purchase_intent_probability"])
    price_sensitivity_level = consumer_intelligence_output["price_sensitivity_level"]
    high_propensity = purchase_probability >= PURCHASE_PROPENSITY_THRESHOLD

    if intent_level == "high" and high_propensity:
        recommended_action = "conversion"
        decision_reason = "High current behavioral intent and ML purchase propensity meet the conversion threshold."
    elif high_propensity:
        recommended_action = "re_engagement"
        decision_reason = "ML purchase propensity meets the threshold, but current behavioral intent needs re-engagement."
    elif intent_level in {"high", "medium"}:
        recommended_action = "consideration"
        decision_reason = "Current behavioral intent is active, but ML purchase propensity is below the threshold."
    else:
        recommended_action = "awareness"
        decision_reason = "Current behavioral intent and ML purchase propensity are both below the conversion threshold."

    messaging_strategy = {
        "high": "value_oriented",
        "medium": "balanced",
        "low": "premium_oriented",
        "unknown": "product_focused",
    }.get(price_sensitivity_level, "product_focused")

    return {
        "recommended_action": recommended_action,
        "recommended_category": consumer_intelligence_output.get("primary_interest"),
        "messaging_strategy": messaging_strategy,
        "decision_reason": decision_reason,
    }


def load_catalog_prices():
    """Load the current product-catalog prices for relative behavioral comparison."""
    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT price::double precision FROM products WHERE price IS NOT NULL")
            return [row[0] for row in cursor.fetchall()]


def get_consumer_intelligence(user_id):
    """Combine saved purchase propensity with deterministic behavioral signals."""
    purchase_prediction = purchase_intent.predict_purchase_intent(user_id)
    as_of = datetime.now()
    events = purchase_intent.load_user_events(user_id, as_of)

    interest = calculate_primary_interest(events, as_of)
    current_intent = calculate_current_intent(events, as_of)
    price_sensitivity = calculate_behavioral_price_sensitivity(events, load_catalog_prices())
    threshold = float(purchase_intent.load_model_artifact()["classification_threshold"])
    segment = assign_consumer_segment(
        current_intent["intent_level"],
        price_sensitivity["price_sensitivity_level"],
        purchase_prediction["purchase_intent_probability"],
        threshold,
    )

    return {
        **purchase_prediction,
        **interest,
        **current_intent,
        **price_sensitivity,
        "consumer_segment": segment,
    }
