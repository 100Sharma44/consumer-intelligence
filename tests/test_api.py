"""Dependency-free tests for FastAPI endpoint handlers."""

import unittest
from unittest.mock import patch

from fastapi import HTTPException

from app.main import health, predict
from app.services.purchase_intent import UserNotFoundError


NORMAL_USER_RESPONSE = {
    "user_id": 1,
    "purchase_intent_probability": 0.25,
    "predicted_purchase_intent": 0,
    "intent_level": "medium",
    "primary_interest": "Electronics",
    "interest_score": 75.0,
    "intent_score": 45.0,
    "price_sensitivity_score": 55.0,
    "price_sensitivity_level": "medium",
    "consumer_segment": "active_researcher",
}

NORMAL_USER_DECISION = {
    "recommended_action": "consideration",
    "recommended_category": "Electronics",
    "messaging_strategy": "balanced",
    "decision_reason": "Current behavioral intent is active, but ML purchase propensity is below the threshold.",
}


class ApiEndpointTests(unittest.TestCase):
    @patch("app.main.purchase_intent.load_model_artifact")
    def test_health(self, mocked_load_model):
        self.assertEqual(health(), {"status": "ok", "model_loaded": True})
        mocked_load_model.assert_called_once()

    @patch("app.main.consumer_intelligence.make_advertising_decision")
    @patch("app.main.consumer_intelligence.get_consumer_intelligence")
    def test_predict_normal_user(self, mocked_intelligence, mocked_decision):
        mocked_intelligence.return_value = NORMAL_USER_RESPONSE
        mocked_decision.return_value = NORMAL_USER_DECISION

        response = predict(1)

        self.assertEqual(response["user_id"], 1)
        self.assertEqual(response["primary_interest"], "Electronics")
        self.assertEqual(response["consumer_segment"], "active_researcher")
        self.assertEqual(response["recommended_action"], "consideration")
        mocked_intelligence.assert_called_once_with(1)
        mocked_decision.assert_called_once_with(NORMAL_USER_RESPONSE)

    @patch("app.main.consumer_intelligence.make_advertising_decision")
    @patch("app.main.consumer_intelligence.get_consumer_intelligence")
    def test_predict_high_propensity_low_current_intent_user(self, mocked_intelligence, mocked_decision):
        mocked_intelligence.return_value = {
            **NORMAL_USER_RESPONSE,
            "user_id": 15,
            "purchase_intent_probability": 0.79,
            "predicted_purchase_intent": 1,
            "intent_level": "low",
            "consumer_segment": "latent_high_propensity",
        }
        mocked_decision.return_value = {
            "recommended_action": "re_engagement",
            "recommended_category": "Electronics",
            "messaging_strategy": "balanced",
            "decision_reason": "ML purchase propensity meets the threshold, but current behavioral intent needs re-engagement.",
        }

        response = predict(15)

        self.assertEqual(response["consumer_segment"], "latent_high_propensity")
        self.assertEqual(response["intent_level"], "low")
        self.assertEqual(response["recommended_action"], "re_engagement")

    @patch("app.main.consumer_intelligence.make_advertising_decision")
    @patch("app.main.consumer_intelligence.get_consumer_intelligence")
    def test_predict_user_with_no_events(self, mocked_intelligence, mocked_decision):
        mocked_intelligence.return_value = {
            **NORMAL_USER_RESPONSE,
            "user_id": 501,
            "intent_level": "low",
            "primary_interest": None,
            "interest_score": 0.0,
            "price_sensitivity_score": 0.0,
            "price_sensitivity_level": "unknown",
            "consumer_segment": "low_intent_browser",
        }
        mocked_decision.return_value = {
            "recommended_action": "awareness",
            "recommended_category": None,
            "messaging_strategy": "product_focused",
            "decision_reason": "Current behavioral intent and ML purchase propensity are both below the conversion threshold.",
        }

        response = predict(501)

        self.assertIsNone(response["primary_interest"])
        self.assertEqual(response["price_sensitivity_level"], "unknown")
        self.assertEqual(response["messaging_strategy"], "product_focused")

    @patch("app.main.consumer_intelligence.get_consumer_intelligence")
    def test_predict_nonexistent_user_returns_404(self, mocked_intelligence):
        mocked_intelligence.side_effect = UserNotFoundError("User 999999 does not exist.")

        with self.assertRaises(HTTPException) as raised:
            predict(999999)

        self.assertEqual(raised.exception.status_code, 404)
        self.assertEqual(raised.exception.detail, "User 999999 does not exist.")


if __name__ == "__main__":
    unittest.main()
