"""Focused unit tests for deterministic V1 advertising decisions."""

import unittest

from app.services.consumer_intelligence import make_advertising_decision


def intelligence_output(intent_level, probability, price_level, primary_interest="Electronics"):
    return {
        "intent_level": intent_level,
        "purchase_intent_probability": probability,
        "price_sensitivity_level": price_level,
        "primary_interest": primary_interest,
    }


class AdvertisingDecisionTests(unittest.TestCase):
    def test_high_intent_high_propensity_recommends_conversion_and_value_message(self):
        decision = make_advertising_decision(intelligence_output("high", 0.80, "high"))

        self.assertEqual(decision["recommended_action"], "conversion")
        self.assertEqual(decision["recommended_category"], "Electronics")
        self.assertEqual(decision["messaging_strategy"], "value_oriented")
        self.assertIn("conversion threshold", decision["decision_reason"])

    def test_medium_intent_high_propensity_recommends_re_engagement_and_balanced_message(self):
        decision = make_advertising_decision(intelligence_output("medium", 0.70, "medium"))

        self.assertEqual(decision["recommended_action"], "re_engagement")
        self.assertEqual(decision["messaging_strategy"], "balanced")

    def test_high_intent_low_propensity_recommends_consideration_and_premium_message(self):
        decision = make_advertising_decision(intelligence_output("high", 0.40, "low"))

        self.assertEqual(decision["recommended_action"], "consideration")
        self.assertEqual(decision["messaging_strategy"], "premium_oriented")

    def test_low_intent_low_propensity_recommends_awareness_and_product_message(self):
        decision = make_advertising_decision(intelligence_output("low", 0.20, "unknown", None))

        self.assertEqual(decision["recommended_action"], "awareness")
        self.assertIsNone(decision["recommended_category"])
        self.assertEqual(decision["messaging_strategy"], "product_focused")


if __name__ == "__main__":
    unittest.main()
