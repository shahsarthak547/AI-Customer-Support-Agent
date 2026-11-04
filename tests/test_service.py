from unittest.mock import patch

from app.api.service import SupportAgentService


def test_unsafe_generated_response_is_escalated():
    service = SupportAgentService()

    with (
        patch.object(service, "load"),
        patch.object(
            service,
            "classify",
            return_value=("battery_charging", 0.99),
        ),
        patch.object(
            service,
            "retrieve",
            return_value=[
                {
                    "customer_text": "My battery drains quickly",
                    "support_text": "Restart your device",
                    "similarity": 0.90,
                }
            ],
        ),
        patch(
            "app.api.service.generate_response",
            return_value="Visit https://example.com",
        ),
    ):
        result = service.process("My battery drains quickly")

    assert result["decision"] == "human_escalation"
    assert result["response"] is None
    assert "URL" in result["decision_reason"]