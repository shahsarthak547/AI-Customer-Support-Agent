
from src.agent.grounding_validator import validate_grounding


KNOWLEDGE = [
    {
        "id": "APPLE-BAT-001",
        "product": "iPhone",
        "intent": "battery_charging",
        "topic": "Check battery usage",
        "guidance": [
            "Open Settings and tap Battery.",
            "Review the battery usage information and identify apps or system activities consuming battery.",
            "Check whether an app has unusually high background activity.",
            "Consider whether low cellular signal or notifications contribute to battery usage.",
        ],
        "verification_status": "verified",
    }
]


def test_supported_battery_usage_instructions():
    response = (
        "Open Settings and tap Battery. "
        "Review the battery usage information and identify "
        "apps or system activities consuming battery."
    )

    result = validate_grounding(response, KNOWLEDGE)

    assert result["grounded"] is True


def test_unsupported_settings_instruction():
    response = (
        "Open Settings and tap Battery. "
        "Go to Settings > General > iPhone Storage "
        "and clear the battery cache."
    )

    result = validate_grounding(response, KNOWLEDGE)

    assert result["grounded"] is False
    assert result["unsupported_sentences"]


def test_empty_response_fails():
    result = validate_grounding("", KNOWLEDGE)

    assert result["grounded"] is False


def test_missing_knowledge_fails():
    result = validate_grounding(
        "Open Settings and tap Battery.",
        [],
    )

    assert result["grounded"] is False


def test_unverified_knowledge_is_not_used():
    unverified = [
        {
            "guidance": [
                "Open Settings and tap Battery."
            ],
            "verification_status": "pending_review",
        }
    ]

    result = validate_grounding(
        "Open Settings and tap Battery.",
        unverified,
    )

    assert result["grounded"] is False
