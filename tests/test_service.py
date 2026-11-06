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
        patch.object(
            service,
            "retrieve_knowledge",
            return_value=[
                {
                    "id": "APPLE-BAT-001",
                    "product": "iPhone",
                    "intent": "battery_charging",
                    "topic": "Battery usage",
                    "guidance": ["Review battery usage in Settings."],
                    "source": "https://support.apple.com/en-in/120745",
                    "similarity": 0.90,
                }
            ],
        ),
    ):
        result = service.process("My battery drains quickly")

    assert result["decision"] == "human_escalation"
    assert result["response"] is None
    assert "URL" in result["decision_reason"]
def test_unsupported_generated_response_is_escalated():
    service = SupportAgentService()

    verified_knowledge = [
        {
            "id": "APPLE-BAT-001",
            "product": "iPhone",
            "intent": "battery_charging",
            "topic": "Check battery usage",
            "guidance": [
                "Open Settings and tap Battery.",
                "Review the battery usage information.",
            ],
            "source": "Apple Support",
            "verification_status": "verified",
        }
    ]

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
                    "customer_text": "Battery drains quickly",
                    "support_text": "Check battery usage",
                    "similarity": 0.90,
                }
            ],
        ),
        patch.object(
            service,
            "retrieve_knowledge",
            return_value=verified_knowledge,
        ),
        patch(
            "app.api.service.generate_response",
            return_value=(
                "Open Settings and tap Battery. "
                "Go to iPhone Storage and clear the battery cache."
            ),
        ),
    ):
        result = service.process(
            "My iPhone battery drains quickly."
        )

    assert result["decision"] == "human_escalation"
    assert result["response"] is None
    assert "grounding" in result["decision_reason"].lower() or (
        "technical statements" in result["decision_reason"].lower()
    )