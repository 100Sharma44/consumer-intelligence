"""Unit tests for deterministic consumer-intelligence rules."""

from datetime import datetime, timedelta
import unittest

import pandas as pd

from app.services.consumer_intelligence import (
    assign_consumer_segment,
    calculate_behavioral_price_sensitivity,
    calculate_current_intent,
    calculate_primary_interest,
)


NOW = datetime(2026, 9, 13, 12, 0, 0)
CATALOG_PRICES = [4999, 6999, 8999, 12999]


def events(rows):
    return pd.DataFrame(rows, columns=["product_id", "event_type", "event_time", "category", "price"])


class ConsumerIntelligenceRuleTests(unittest.TestCase):
    def test_primary_interest_favors_recent_high_engagement_category(self):
        user_events = events(
            [
                (1, "view", NOW - timedelta(days=40), "Running Shoes", 8999),
                (2, "cart", NOW - timedelta(days=1), "Electronics", 6999),
                (2, "wishlist", NOW - timedelta(days=2), "Electronics", 6999),
            ]
        )

        result = calculate_primary_interest(user_events, NOW)

        self.assertEqual(result["primary_interest"], "Electronics")
        self.assertGreater(result["interest_score"], 80)

    def test_current_intent_rewards_recent_cart_and_wishlist(self):
        user_events = events(
            [
                (1, "search", NOW - timedelta(days=45), "Running Shoes", 8999),
                (2, "wishlist", NOW - timedelta(hours=2), "Electronics", 6999),
                (2, "cart", NOW - timedelta(hours=1), "Electronics", 6999),
            ]
        )

        result = calculate_current_intent(user_events, NOW)

        self.assertEqual(result["intent_level"], "high")
        self.assertGreaterEqual(result["intent_score"], 60)

    def test_behavioral_price_sensitivity_is_high_for_lower_observed_prices(self):
        user_events = events(
            [
                (4, "view", NOW, "Clothing", 4999),
                (4, "cart", NOW, "Clothing", 4999),
            ]
        )

        result = calculate_behavioral_price_sensitivity(user_events, CATALOG_PRICES)

        self.assertEqual(result["price_sensitivity_level"], "high")
        self.assertGreaterEqual(result["price_sensitivity_score"], 65)

    def test_behavioral_price_sensitivity_is_low_for_higher_observed_prices(self):
        user_events = events(
            [
                (2, "view", NOW, "Running Shoes", 12999),
                (2, "cart", NOW, "Running Shoes", 12999),
            ]
        )

        result = calculate_behavioral_price_sensitivity(user_events, CATALOG_PRICES)

        self.assertEqual(result["price_sensitivity_level"], "low")
        self.assertLess(result["price_sensitivity_score"], 35)

    def test_behavioral_price_sensitivity_is_unknown_without_catalog_prices(self):
        user_events = events([(1, "view", NOW, "Clothing", 4999)])

        result = calculate_behavioral_price_sensitivity(user_events, [])

        self.assertEqual(
            result,
            {"price_sensitivity_score": 0.0, "price_sensitivity_level": "unknown"},
        )

    def test_no_events_returns_clean_unknown_or_low_signals(self):
        result_interest = calculate_primary_interest(events([]), NOW)
        result_intent = calculate_current_intent(events([]), NOW)
        result_price = calculate_behavioral_price_sensitivity(events([]), CATALOG_PRICES)

        self.assertEqual(result_interest, {"primary_interest": None, "interest_score": 0.0})
        self.assertEqual(result_intent, {"intent_score": 0.0, "intent_level": "low"})
        self.assertEqual(
            result_price,
            {"price_sensitivity_score": 0.0, "price_sensitivity_level": "unknown"},
        )

    def test_high_current_intent_and_high_propensity_gets_high_intent_segment(self):
        threshold = 0.62
        self.assertEqual(
            assign_consumer_segment("high", "high", 0.80, threshold),
            "high_intent_value_seeker",
        )
        self.assertEqual(
            assign_consumer_segment("high", "low", 0.80, threshold),
            "high_intent_premium_shopper",
        )

    def test_high_current_intent_and_low_propensity_is_active_researcher(self):
        self.assertEqual(
            assign_consumer_segment("high", "medium", 0.20, 0.62),
            "active_researcher",
        )

    def test_low_current_intent_and_high_propensity_is_latent_high_propensity(self):
        self.assertEqual(
            assign_consumer_segment("low", "medium", 0.80, 0.62),
            "latent_high_propensity",
        )

    def test_low_current_intent_and_low_propensity_is_low_intent_browser(self):
        self.assertEqual(
            assign_consumer_segment("low", "unknown", 0.20, 0.62),
            "low_intent_browser",
        )


if __name__ == "__main__":
    unittest.main()
